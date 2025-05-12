from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, date, timedelta
import os
import cv2
import face_recognition
import numpy as np
import pandas as pd 
from io import BytesIO
import pytz
from flask_wtf.csrf import CSRFProtect
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///attendance.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Khởi tạo CSRF protection
csrf = CSRFProtect(app)

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    employee_id = db.Column(db.String(20), unique=True, nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    date_of_birth = db.Column(db.Date, nullable=False)
    department = db.Column(db.String(50), nullable=False)
    face_encoding = db.Column(db.PickleType, nullable=True)
    is_admin = db.Column(db.Boolean, default=False)
    attendance_records = db.relationship('Attendance', backref='user', lazy=True)

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    check_in_time = db.Column(db.DateTime, nullable=False)
    status = db.Column(db.String(20), default='Present')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            if user.is_admin:
                return redirect(url_for('admin_dashboard'))
            return redirect(url_for('employee_dashboard'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        return redirect(url_for('employee_dashboard'))
    
    active_tab = request.args.get('active_tab', 'attendance')
    vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
    
    # Xử lý lọc cho bản ghi điểm danh
    date_filter = request.args.get('date')
    month_filter = request.args.get('month')
    employee_id_filter = request.args.get('employee_id')
    department_filter = request.args.get('department')
    
    # Query cho bản ghi điểm danh với bộ lọc
    attendance_query = Attendance.query.join(User)
    
    if date_filter:
        attendance_query = attendance_query.filter(db.func.date(Attendance.check_in_time) == datetime.strptime(date_filter, '%Y-%m-%d').date())
    elif month_filter:
        month_date = datetime.strptime(month_filter, '%Y-%m')
        attendance_query = attendance_query.filter(
            db.func.extract('year', Attendance.check_in_time) == month_date.year,
            db.func.extract('month', Attendance.check_in_time) == month_date.month
        )
    
    if employee_id_filter:
        attendance_query = attendance_query.filter(User.employee_id == employee_id_filter)
    
    if department_filter:
        attendance_query = attendance_query.filter(User.department == department_filter)
    
    attendance_records = attendance_query.order_by(Attendance.check_in_time.desc()).all()
    
    # Chuyển đổi thời gian sang múi giờ Việt Nam cho tất cả bản ghi
    for record in attendance_records:
        record.check_in_time = record.check_in_time.astimezone(vietnam_tz)

    # Xử lý lọc cho danh sách nhân viên
    employee_search = request.args.get('employee_search')
    employee_department = request.args.get('employee_department')
    employee_month = request.args.get('employee_month')
    
    employee_query = User.query.filter_by(is_admin=False)
    
    if employee_search:
        employee_query = employee_query.filter(User.employee_id.ilike(f'%{employee_search}%'))
    
    if employee_department:
        employee_query = employee_query.filter(User.department == employee_department)
    
    employees = employee_query.all()

    # Tính số buổi điểm danh muộn cho từng nhân viên
    for employee in employees:
        # Ưu tiên lọc theo employee_month nếu có
        if employee_month:
            try:
                month_date = datetime.strptime(employee_month, '%Y-%m')
            except Exception:
                month_date = datetime.now().astimezone(vietnam_tz)
            month_start = month_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            next_month = (month_start + timedelta(days=32)).replace(day=1)
            
            monthly_attendance = Attendance.query.filter(
                Attendance.user_id == employee.id,
                Attendance.check_in_time >= month_start,
                Attendance.check_in_time < next_month
            ).all()
            late_count = sum(1 for record in monthly_attendance 
                           if record.check_in_time.astimezone(vietnam_tz).hour >= 9)
            employee.late_count = late_count
            employee.total_attendance = len(monthly_attendance)
        elif month_filter:
            month_date = datetime.strptime(month_filter, '%Y-%m')
            month_start = month_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            next_month = (month_start + timedelta(days=32)).replace(day=1)
            
            monthly_attendance = Attendance.query.filter(
                Attendance.user_id == employee.id,
                Attendance.check_in_time >= month_start,
                Attendance.check_in_time < next_month
            ).all()
            late_count = sum(1 for record in monthly_attendance 
                           if record.check_in_time.astimezone(vietnam_tz).hour >= 9)
            employee.late_count = late_count
            employee.total_attendance = len(monthly_attendance)
        else:
            # Lấy toàn bộ bản ghi điểm danh của nhân viên đó từ trước đến nay
            all_attendance = Attendance.query.filter(
                Attendance.user_id == employee.id
            ).all()
            late_count = sum(1 for record in all_attendance 
                           if record.check_in_time.astimezone(vietnam_tz).hour >= 9)
            employee.late_count = late_count
            employee.total_attendance = len(all_attendance)

    departments = db.session.query(User.department).distinct().all()
    departments = [dept[0] for dept in departments if dept[0]]

    current_filters = {
        'date': date_filter,
        'month': month_filter,
        'employee_id': employee_id_filter,
        'department': department_filter
    }

    employee_filters = {
        'search': employee_search,
        'department': employee_department,
        'month': employee_month
    }
    
    return render_template('admin_dashboard.html',
                         active_tab=active_tab,
                         employees=employees,
                         attendance_records=attendance_records,
                         departments=departments,
                         current_filters=current_filters,
                         employee_filters=employee_filters)

@app.route('/employee/dashboard')
@login_required
def employee_dashboard():
    # Lấy múi giờ Việt Nam
    vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
    now = datetime.now().astimezone(vietnam_tz)
    today = now.date()
    
    # Lấy tham số lọc
    date_filter = request.args.get('date')
    month_filter = request.args.get('month')
    
    # Query điểm danh hôm nay
    today_attendance = Attendance.query.filter(
        db.func.date(Attendance.check_in_time) == today,
        Attendance.user_id == current_user.id
    ).first()
    
    # Query lịch sử điểm danh với bộ lọc
    query = Attendance.query.filter_by(user_id=current_user.id)
    
    if date_filter:
        query = query.filter(db.func.date(Attendance.check_in_time) == datetime.strptime(date_filter, '%Y-%m-%d').date())
    elif month_filter:
        month_date = datetime.strptime(month_filter, '%Y-%m')
        query = query.filter(
            db.func.extract('year', Attendance.check_in_time) == month_date.year,
            db.func.extract('month', Attendance.check_in_time) == month_date.month
        )
    
    attendance_records = query.order_by(Attendance.check_in_time.desc()).all()
    
    # Chuyển đổi thời gian sang múi giờ Việt Nam
    if today_attendance:
        today_attendance.check_in_time = today_attendance.check_in_time.astimezone(vietnam_tz)
    
    for record in attendance_records:
        record.check_in_time = record.check_in_time.astimezone(vietnam_tz)
    
    # Xác định tháng thống kê: ưu tiên month_filter, nếu không có thì lấy tháng hiện tại
    if month_filter:
        stat_month_date = datetime.strptime(month_filter, '%Y-%m')
    else:
        stat_month_date = now
    stat_month_start = stat_month_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    next_month = (stat_month_start + timedelta(days=32)).replace(day=1)
    
    monthly_attendance = Attendance.query.filter(
        Attendance.user_id == current_user.id,
        Attendance.check_in_time >= stat_month_start,
        Attendance.check_in_time < next_month
    ).all()
    
    late_days = sum(1 for record in monthly_attendance if record.check_in_time.astimezone(vietnam_tz).hour >= 9)
    total_days = len(monthly_attendance)
    
    # Định dạng tháng theo dạng số/tháng/năm
    stat_month_display = f"{stat_month_start.month}/{stat_month_start.year}"
    
    current_filters = {
        'date': date_filter,
        'month': month_filter
    }
    
    return render_template('employee_dashboard.html',
                         today_attendance=today_attendance,
                         attendance_records=attendance_records,
                         current_filters=current_filters,
                         late_days=late_days,
                         total_days=total_days,
                         stat_month_display=stat_month_display,
                         stat_month_value=stat_month_start.strftime('%Y-%m'))

@app.route('/face_recognition')
@login_required
def face_recognition_page():
    return render_template('face_recognition.html')

@app.route('/check_attendance', methods=['POST'])
@login_required
def check_attendance():
    # Kiểm tra xem có ảnh không
    if 'image' not in request.json:
        return jsonify({'error': 'Không có ảnh được gửi lên'}), 400
    
    # Lấy múi giờ Việt Nam và thời gian hiện tại
    vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
    now = datetime.now().astimezone(vietnam_tz)
    today = now.date()
    
    # Kiểm tra đã điểm danh hôm nay chưa
    existing_attendance = Attendance.query.filter(
        db.func.date(Attendance.check_in_time) == today,
        Attendance.user_id == current_user.id
    ).first()
    
    if existing_attendance:
        local_time = existing_attendance.check_in_time.astimezone(vietnam_tz)
        return jsonify({
            'error': f'Bạn đã điểm danh hôm nay lúc {local_time.strftime("%H:%M:%S")}'
        }), 400
    
    try:
        # Chuyển đổi base64 thành numpy array
        image_data = request.json['image'].split(',')[1]
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Tìm khuôn mặt trong ảnh
        face_locations = face_recognition.face_locations(rgb_img)
        if not face_locations:
            return jsonify({'error': 'Không phát hiện khuôn mặt trong ảnh'}), 400
            
        # Kiểm tra chỉ có một khuôn mặt trong ảnh
        if len(face_locations) > 1:
            return jsonify({'error': 'Chỉ được phép có một khuôn mặt trong ảnh'}), 400
        
        # Lấy encoding của khuôn mặt
        face_encoding = face_recognition.face_encodings(rgb_img, face_locations)[0]
        
        # So sánh với khuôn mặt đã lưu
        user_face_encoding = current_user.face_encoding
        if user_face_encoding is None:
            return jsonify({'error': 'Không tìm thấy dữ liệu khuôn mặt của bạn'}), 400
        
        # Kiểm tra với khuôn mặt đã lưu với ngưỡng thấp hơn
        matches = face_recognition.compare_faces([user_face_encoding], face_encoding, tolerance=0.4)
        if not any(matches):
            return jsonify({'error': 'Không nhận diện được khuôn mặt. Vui lòng thử lại'}), 400
            
        # Tính toán khoảng cách giữa các khuôn mặt
        face_distances = face_recognition.face_distance([user_face_encoding], face_encoding)
        if face_distances[0] > 0.4:  # Nếu khoảng cách lớn hơn 0.4 thì không chấp nhận
            return jsonify({'error': 'Khuôn mặt không khớp. Vui lòng thử lại'}), 400
        
        # Lấy thời gian hiện tại theo múi giờ Việt Nam
        check_in_time = datetime.now().astimezone(vietnam_tz)
        
        # Ghi nhận điểm danh
        attendance = Attendance(
            user_id=current_user.id,
            check_in_time=check_in_time
        )
        db.session.add(attendance)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Đã ghi nhận điểm danh lúc {check_in_time.strftime("%H:%M:%S")}'
        })
        
    except Exception as e:
        print(f"Error in check_attendance: {str(e)}")
        return jsonify({'error': 'Có lỗi xảy ra khi xử lý ảnh'}), 500

@app.route('/add_employee', methods=['POST'])
@login_required
def add_employee():
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403

    # Kiểm tra các trường bắt buộc
    required_fields = ['username', 'password', 'employee_id', 'full_name', 'date_of_birth', 'department']
    for field in required_fields:
        if not request.form.get(field):
            flash(f'Vui lòng điền đầy đủ thông tin cho trường {field}', 'error')
            return redirect(url_for('admin_dashboard'))

    # Kiểm tra file ảnh
    if 'face_image' not in request.files:
        flash('Vui lòng chọn ảnh khuôn mặt', 'error')
        return redirect(url_for('admin_dashboard'))

    face_image = request.files['face_image']
    if face_image.filename == '':
        flash('Vui lòng chọn ảnh khuôn mặt', 'error')
        return redirect(url_for('admin_dashboard'))

    try:
        # Kiểm tra username và employee_id đã tồn tại chưa
        if User.query.filter_by(username=request.form['username']).first():
            flash('Tên đăng nhập đã tồn tại', 'error')
            return redirect(url_for('admin_dashboard'))
        
        if User.query.filter_by(employee_id=request.form['employee_id']).first():
            flash('Mã nhân viên đã tồn tại', 'error')
            return redirect(url_for('admin_dashboard'))

        # Xử lý ảnh khuôn mặt
        img = face_recognition.load_image_file(face_image)
        face_locations = face_recognition.face_locations(img)
        
        if not face_locations:
            flash('Không phát hiện khuôn mặt trong ảnh', 'error')
            return redirect(url_for('admin_dashboard'))
            
        if len(face_locations) > 1:
            flash('Chỉ được phép có một khuôn mặt trong ảnh', 'error')
            return redirect(url_for('admin_dashboard'))

        face_encoding = face_recognition.face_encodings(img, face_locations)[0]

        # Kiểm tra khuôn mặt trùng lặp với các nhân viên khác
        existing_users = User.query.filter(User.face_encoding.isnot(None)).all()
        for user in existing_users:
            if user.face_encoding is not None:
                # So sánh khuôn mặt với ngưỡng thấp hơn để phát hiện trùng lặp
                matches = face_recognition.compare_faces([user.face_encoding], face_encoding, tolerance=0.4)
                if any(matches):
                    flash('Khuôn mặt này đã được đăng ký cho nhân viên khác. Vui lòng sử dụng khuôn mặt khác.', 'error')
                    return redirect(url_for('admin_dashboard'))

        # Tạo user mới
        new_user = User(
            username=request.form['username'],
            password_hash=generate_password_hash(request.form['password']),
            employee_id=request.form['employee_id'],
            full_name=request.form['full_name'],
            date_of_birth=datetime.strptime(request.form['date_of_birth'], '%Y-%m-%d').date(),
            department=request.form['department'],
            face_encoding=face_encoding,
            is_admin=False
        )

        db.session.add(new_user)
        db.session.commit()
        flash('Thêm nhân viên thành công', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash(f'Có lỗi xảy ra: {str(e)}', 'error')
        
    return redirect(url_for('admin_dashboard'))

@app.route('/delete_employee/<int:employee_id>', methods=['POST'])
@login_required
def delete_employee(employee_id):
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Không có quyền thực hiện'}), 403
    
    try:
        # Lấy thông tin nhân viên
        employee = User.query.get_or_404(employee_id)
        
        # Kiểm tra không cho phép xóa tài khoản admin
        if employee.is_admin:
            return jsonify({'success': False, 'error': 'Không thể xóa tài khoản admin'}), 400
        
        # Xóa các bản ghi điểm danh của nhân viên
        Attendance.query.filter_by(user_id=employee_id).delete()
        
        # Xóa nhân viên
        db.session.delete(employee)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Đã xóa nhân viên thành công'})
        
    except Exception as e:
        db.session.rollback()
        print(f"Error deleting employee: {str(e)}")
        return jsonify({'success': False, 'error': 'Lỗi khi xóa nhân viên'}), 500

@app.route('/export_attendance')
@login_required
def export_attendance():
    # Lấy múi giờ Việt Nam
    vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
    
    if not current_user.is_admin:
        # Đối với người dùng không phải admin, chỉ hiển thị bản ghi của chính họ
        attendance_records = Attendance.query.filter_by(user_id=current_user.id).all()
    else:
        # Đối với admin, áp dụng bộ lọc tương tự như bảng điều khiển
        query = Attendance.query.join(User)
        
        date_filter = request.args.get('date')
        month_filter = request.args.get('month')
        employee_id_filter = request.args.get('employee_id')
        department_filter = request.args.get('department')
        
        if date_filter:
            query = query.filter(db.func.date(Attendance.check_in_time) == datetime.strptime(date_filter, '%Y-%m-%d').date())
        elif month_filter:
            month_date = datetime.strptime(month_filter, '%Y-%m')
            query = query.filter(
                db.func.extract('year', Attendance.check_in_time) == month_date.year,
                db.func.extract('month', Attendance.check_in_time) == month_date.month
            )
        
        if employee_id_filter:
            query = query.filter(User.employee_id == employee_id_filter)
        
        if department_filter:
            query = query.filter(User.department == department_filter)
            
        attendance_records = query.order_by(Attendance.check_in_time.desc()).all()
    
    # Tạo DataFrame
    data = []
    for record in attendance_records:
        # Chuyển đổi thời gian sang múi giờ Việt Nam
        local_time = record.check_in_time.astimezone(vietnam_tz)
        
        user = User.query.get(record.user_id)
        data.append({
            'Mã nhân viên': user.employee_id,
            'Họ tên': user.full_name,
            'Phòng ban': user.department,
            'Ngày': local_time.strftime('%d/%m/%Y'),
            'Thời gian': local_time.strftime('%H:%M:%S'),
            'Trạng thái': 'Có mặt'
        })
    
    df = pd.DataFrame(data)
    
    # Tạo file Excel trong bộ nhớ
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Điểm danh')
    
    output.seek(0)
    
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='diem_danh.xlsx'
    )

@app.route('/get_employee/<int:employee_id>')
@login_required
def get_employee(employee_id):
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
    
    employee = User.query.get_or_404(employee_id)
    return jsonify({
        'id': employee.id,
        'username': employee.username,
        'employee_id': employee.employee_id,
        'full_name': employee.full_name,
        'date_of_birth': employee.date_of_birth.strftime('%Y-%m-%d'),
        'department': employee.department
    })

@app.route('/update_employee/<int:employee_id>', methods=['POST'])
@login_required
def update_employee(employee_id):
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
    
    employee = User.query.get_or_404(employee_id)
    
    try:
        # Cập nhật thông tin cơ bản
        employee.username = request.form.get('username')
        # Không cho phép thay đổi mã nhân viên
        # employee.employee_id = request.form.get('employee_id')
        employee.full_name = request.form.get('full_name')
        employee.date_of_birth = datetime.strptime(request.form.get('date_of_birth'), '%Y-%m-%d').date()
        employee.department = request.form.get('department')
        
        # Cập nhật mật khẩu nếu cung cấp
        new_password = request.form.get('password')
        if new_password and new_password.strip():
            employee.password_hash = generate_password_hash(new_password)
        
        # Cập nhật ảnh khuôn mặt nếu cung cấp
        face_image = request.files.get('face_image')
        if face_image:
            img = cv2.imdecode(np.fromstring(face_image.read(), np.uint8), cv2.IMREAD_COLOR)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_img)
            
            if not face_locations:
                return jsonify({'error': 'No face detected in the image'}), 400
            
            face_encoding = face_recognition.face_encodings(rgb_img, face_locations)[0]

            # Kiểm tra khuôn mặt trùng lặp với các nhân viên khác
            existing_users = User.query.filter(
                User.face_encoding.isnot(None),
                User.id != employee_id  # Loại trừ nhân viên hiện tại
            ).all()
            for user in existing_users:
                if user.face_encoding is not None:
                    # So sánh khuôn mặt với ngưỡng thấp hơn để phát hiện trùng lặp
                    matches = face_recognition.compare_faces([user.face_encoding], face_encoding, tolerance=0.4)
                    if any(matches):
                        return jsonify({'error': 'Khuôn mặt này đã được đăng ký cho nhân viên khác. Vui lòng sử dụng khuôn mặt khác.'}), 400
            
            employee.face_encoding = face_encoding
        
        db.session.commit()
        return jsonify({'success': True, 'message': 'Employee updated successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400

@app.route('/export_employees')
@login_required
def export_employees():
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
    
    # Lấy tham số tìm kiếm và lọc
    search = request.args.get('search')
    department = request.args.get('department')
    
    # Query danh sách nhân viên với bộ lọc
    query = User.query.filter_by(is_admin=False)
    
    if search:
        query = query.filter(User.employee_id.ilike(f'%{search}%'))
    
    if department:
        query = query.filter(User.department == department)
    
    employees = query.all()
    
    # Tạo DataFrame
    data = []
    for employee in employees:
        data.append({
            'Mã nhân viên': employee.employee_id,
            'Họ tên': employee.full_name,
            'Tên đăng nhập': employee.username,
            'Phòng ban': employee.department,
            'Ngày sinh': employee.date_of_birth.strftime('%Y-%m-%d')
        })
    
    df = pd.DataFrame(data)
    
    # Tạo file Excel trong bộ nhớ
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Danh sách nhân viên')
    
    output.seek(0)
    
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='danh_sach_nhan_vien.xlsx'
    )

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # Tạo tài khoản admin nếu không tồn tại
        if not User.query.filter_by(username='admin').first():
            admin = User(
                username='admin',
                password_hash=generate_password_hash('admin123'),
                employee_id='ADMIN001',
                full_name='Administrator',
                date_of_birth=datetime(2000, 1, 1),
                department='Administration',
                is_admin=True
            )
            db.session.add(admin)
            db.session.commit()
    app.run() 
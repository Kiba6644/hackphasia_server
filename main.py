from flask import Flask, request, jsonify, send_from_directory, Response
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from datetime import datetime
from libretranslatepy import LibreTranslateAPI
from sqlalchemy.orm import class_mapper
import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER_MICRO = 'uploads/microcourses'
UPLOAD_FOLDER_MAIN = 'uploads/maincourses'
app = Flask(__name__)
app.config['SECRET_KEY'] = 'dasd'

lt = LibreTranslateAPI("https://libretranslate.de/")
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
os.makedirs(UPLOAD_FOLDER_MICRO, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_MAIN, exist_ok=True)
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    password = db.Column(db.String(120), nullable=False)
    name = db.Column(db.String(100))
    email = db.Column(db.String(120), unique=True)
    phone = db.Column(db.String(10))
    age = db.Column(db.Integer)
    gender = db.Column(db.Boolean)  
    preferred_language = db.Column(db.Integer)
    badges = db.Column(db.JSON) 
    certificates = db.Column(db.JSON) 
    posts = db.Column(db.JSON) 
    attendance = db.Column(db.Integer, default=10)
    streaks = db.Column(db.Integer, default=7)
    prev_date = db.Column(db.Date)

    def __repr__(self):
        return f"<User {self.username}>"

class MicroCourse(db.Model):
    course_id = db.Column(db.Integer, primary_key=True)
    video_path = db.Column(db.String(255), nullable=False)
    video_title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(500), nullable=False)
    date_uploaded = db.Column(db.Date, default=datetime.utcnow)

    def __repr__(self):
        return f"<MicroCourse {self.video_title}>"

class MainCourse(db.Model):
    course_id = db.Column(db.Integer, primary_key=True)
    video_path = db.Column(db.String(255), nullable=False)
    video_title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(500), nullable=False)
    date_uploaded = db.Column(db.Date, default=datetime.utcnow)

    def __repr__(self):
        return f"<MainCourse {self.video_title}>"

with app.app_context():
    db.create_all()

class otherfunc():
    def get_video_range(file_path, byte_range):
        with open(file_path, 'rb') as f:
            f.seek(byte_range[0])
            return f.read(byte_range[1] - byte_range[0])

    def calculate_user_score(user):
        badge_score = sum(badge['score'] for badge in user.badges) if user.badges else 0
        certificate_score = sum(certificate['score'] for certificate in user.certificates) if user.certificates else 0
        streak_score = user.streaks * 1.5
        attendance_score = user.attendance
        posts_score = len(user.posts) * 50
        total_score = badge_score + certificate_score + streak_score + attendance_score + posts_score
        
        return total_score

    def avl_courses():
            micro_courses = MicroCourse.query.all()
            main_courses = MainCourse.query.all()
            micro_course_list = [{
                'title': course.video_title,
                'description': course.description,
                'date_uploaded': course.date_uploaded
            } for course in micro_courses]

            main_course_list = [{
                'title': course.video_title,
                'description': course.description,
                'date_uploaded': course.date_uploaded
            } for course in main_courses]

            all_courses = micro_course_list + main_course_list
            random.shuffle(all_courses)

            return all_courses

@app.route('/')
def home():
    data = request.get_json()
    name = data.get('name')
    user = User.query.filter_by(name=name).first()
    if not user:
        return jsonify({"message": "User not found"}), 404
    info = {}
    user_info = {}
    for column in class_mapper(User).columns:
        if column.name not in ['certificates', 'badges', 'streaks']:
            user_info[column.name] = getattr(user, column.name)

    info['users'] = user_info
    info['certificates'] = user.certificates if user.certificates else []
    info['streak'] = user.streaks
    info['courses'] = otherfunc.all_courses

    return jsonify(info), 200

@app.route('/translate')
def translate():
    data = request.get_json()
    name = data.get('name')
    user = User.query.filter_by(name=name).first()
    return jsonify(lt.translate(data.get('text'), target_lang=user.preferred_language))

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()

    if not data or not data.get('name') or not data.get('password'):
        return jsonify({"message": "name and password are required"}), 400

    password = data.get('password')
    name = data.get('name')
    email = data.get('email')
    phone = data.get('phone')
    age = data.get('age')
    gender = data.get('gender')
    preferred_language = data.get('preferred_language')

    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        return jsonify({"message": "Username already exists"}), 400

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(
        username=username,
        password=hashed_password,
        name=name,
        email=email,
        phone=phone,
        age=age,
        gender=gender,
        preferred_language=preferred_language,,
        attendance=0,
        streaks=0
    )
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "User registered successfully"}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()

    if not data or not data.get('username') or not data.get('password'):
        return jsonify({"message": "Username and password are required"}), 400

    email = data.get('email')
    password = data.get('password')

    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({"message": "User not found"}), 404

    if not bcrypt.check_password_hash(user.password, password):
        return jsonify({"message": "Invalid password"}), 401

    return jsonify({"message": "Login successful", "username": user.username}), 200
    
@app.route('/upload_course', methods=['POST'])
def upload_course():
    data = request.form
    name = data.get('name')
    course_type = data.get('course_type')
    video_title = data.get('video_title')
    description = data.get('description')
    user = User.query.filter_by(name=name).first()
    if not course_type or course_type not in ['micro', 'main']:
        return jsonify({"message": "Invalid course type. Choose 'micro' or 'main'."}), 400

    if 'file' not in request.files:
        return jsonify({"message": "No file part."}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"message": "No selected file."}), 400
    
    if file.filename.lower().endswith('.mp4'):
        course_id = db.session.query(db.func.max(MicroCourse.course_id if course_type == 'micro' else MainCourse.course_id)).scalar() + 1
        
        filename = secure_filename(f"{course_id}.mp4")
        
        if course_type == 'micro':
            file.save(os.path.join(UPLOAD_FOLDER_MICRO, filename))
            new_course = MicroCourse(
                course_id=course_id,
                video_path=os.path.join(UPLOAD_FOLDER_MICRO, filename),
                video_title=video_title,
                description=description
            )
        else:
            file.save(os.path.join(UPLOAD_FOLDER_MAIN, filename))
            new_course = MainCourse(
                course_id=course_id,
                video_path=os.path.join(UPLOAD_FOLDER_MAIN, filename),
                video_title=video_title,
                description=description
            )
        db.session.add(new_course)
        db.session.commit()
        new_post = {
            'course_id': course_id,
            'title': video_title,
            'description': description,
            'date_uploaded': datetime.utcnow().strftime('%d-%m-%Y')
        }

        if user.posts:
            user.posts.append(new_post)
        else:
            user.posts = [new_post]
        db.session.commit()
        return jsonify({"message": f"{course_type.capitalize()} course uploaded successfully", "course_id": course_id}), 201

    return jsonify({"message": "File type not allowed. Only MP4 files are allowed."}), 400

@app.route('/stream_course/<int:course_id>', methods=['GET'])
def stream_course(course_id):
    course_type = request.args.get('course_type', 'micro')

    if course_type == 'micro':
        course = MicroCourse.query.filter_by(course_id=course_id).first()
        if not course:
            return jsonify({"message": "Course not found"}), 404
        video_path = os.path.join(UPLOAD_FOLDER_MICRO, os.path.basename(course.video_path))
    else:
        course = MainCourse.query.filter_by(course_id=course_id).first()
        if not course:
            return jsonify({"message": "Course not found"}), 404
        video_path = os.path.join(UPLOAD_FOLDER_MAIN, os.path.basename(course.video_path))

    # Get the file size
    file_size = os.path.getsize(video_path)

    # Get the range from the request (e.g., 'bytes=0-1023')
    range_header = request.headers.get('Range', None)
    if not range_header:
        return jsonify({"message": "Range header missing"}), 400

    # Parse the range header (e.g., 'bytes=0-1023')
    byte_range = range_header.strip().lower().split('=')[1]
    byte_range = byte_range.split(',')
    byte_range = byte_range[0].split('-')  # Handle single range (no multiple ranges)

    start = int(byte_range[0])
    end = int(byte_range[1]) if byte_range[1] else file_size - 1  # If end is not provided, use the file size

    # Handle the case where the range is out of bounds (i.e., larger than the file size)
    if start >= file_size:
        return jsonify({"message": "Requested range is out of bounds"}), 416

    # Get the chunk data
    chunk_data = otherfunc.get_video_range(video_path, (start, end + 1))

    name = request.args.get('name')
    user = User.query.filter_by(name=name).first()

    if user:
        today = datetime.utcnow().date()
        prev_date = user.prev_date
        if prev_date:
            if prev_date == today - timedelta(days=1):
                user.streaks += 1
            elif prev_date < today - timedelta(days=1):
                user.streaks = 0
        else:
            user.streaks = 1
        user.prev_date = today
        db.session.commit()
        response = Response(chunk_data, 206, content_type="video/mp4")
        response.headers.add('Content-Range', f'bytes {start}-{end}/{file_size}')
        response.headers.add('Content-Length', str(len(chunk_data)))
        response.headers.add('Accept-Ranges', 'bytes')

        return response

    return jsonify({"message": "User not found"}), 404

@app.route('/complete_course', methods=['POST'])
def complete_course():
    data = request.get_json()

    username = data.get('username')
    course_id = data.get('course_id')
    course_type = data.get('type')

    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({"message": "User not found"}), 404

    # Get the course by course_id
    if course_type == 'micro':
        course = MicroCourse.query.filter_by(course_id=course_id).first()
    else:
        course = MainCourse.query.filter_by(course_id=course_id).first()
    if not course:
        return jsonify({"message": "Course not found"}), 404

    new_badge = {
        "course_id": course.course_id,
        "course_name": course.video_title,
        "score": 0, 
        "date": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") 
    }

    if user.badges is None:
        user.badges = []
    if user.certificates is None:
        user.certificates = []
    
    if course_type == 'micro':
        user.badges.append(new_badge)
    else:
        user.certificates.append(new_badge)
    db.session.commit()

    return jsonify({"message": "Course completed and badge awarded", "badge": new_badge}), 200

@app.route('/leaderboard', methods=['GET'])
def get_leaderboard():
    users = User.query.all()
    user_scores = []

    for user in users:
        score = calculate_user_score(user)
        user_scores.append({
            'username': user.username,
            'score': score,
            'name': user.name,
            'badges_count': len(user.badges) if user.badges else 0,
            'certificates_count': len(user.certificates) if user.certificates else 0,
            'streaks': user.streaks,
            'attendance': user.attendance
        })

    sorted_user_scores = sorted(user_scores, key=lambda x: x['score'], reverse=True)

    return jsonify(sorted_user_scores), 200


if __name__ == '__main__':
    app.run(debug=True)

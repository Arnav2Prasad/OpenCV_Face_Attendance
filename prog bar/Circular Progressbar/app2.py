from flask import Flask, render_template, request
import os
import cv2
from datetime import date
from PIL import Image
from keras.models import load_model
from PIL import Image
from collections import defaultdict
from imutils.video import VideoStream
import face_recognition

app = Flask(__name__)

nimgs = 10
datetoday2 = date.today().strftime("%d-%B-%Y")
app_running = True

# Load the face recognition model
model = load_model('eye_status_classifier.h5')
model.summary()

# Load the face detection cascade classifier
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Load the eye detection cascade classifiers
open_eyes_detector = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
left_eye_detector = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
right_eye_detector = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')

# Load face encodings
data = np.load('encodings.npy', allow_pickle='TRUE').item()


app = Flask(__name__)

nimgs = 20

imgBackground = cv2.imread("bg.png")

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

if 'students.xlsx' not in os.listdir():
    wb = Workbook()
    ws = wb.active
    ws.append(['Name', 'MIS'])
    wb.save('students.xlsx')

app_running = True

def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l

def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    if userid.strip():  # Check if userid is not an empty string
        try:
            userid_int = int(userid)
            df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
            if userid_int not in df['Roll'].values:
                with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
                    f.write(f'\n{username},{userid_int},{current_time}')
        except ValueError:
            print(f"Invalid userid: {userid}")
    else:
        print("Empty userid provided")

def update_students_excel(name, mis):
    wb = load_workbook('students.xlsx')
    ws = wb.active
    ws.append([name, mis])
    wb.save('students.xlsx')

def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l

@app.route('/')
def new_home():
    names, rolls, times, l = extract_attendance()
    return render_template('new_home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/attendance_page')
def display_attendance_page():
    names, rolls, times, l = extract_attendance()
    return render_template('attendance.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/newuser_page')
def display_newuser_page():
    names, rolls, times, l = extract_attendance()
    return render_template('newuser.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/qr_page')
def display_qr_page():
    names, rolls, times, l = extract_attendance()
    return render_template('qr.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/newuser', methods=['GET', 'POST'])
def newuser():
    if request.method == 'POST':
        username = request.form['username']
        userid = request.form['userid']
        update_students_excel(username, userid)

        userimagefolder = 'static/faces/' + username + '_' + str(userid)
        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)
        i, j = 0, 0
        cap = cv2.VideoCapture(0)
        while 1:
            _, frame = cap.read()
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                if j % 5 == 0:
                    name = username + '_' + str(i) + '.jpg'
                    cv2.imwrite(userimagefolder + '/' + name, frame[y:y + h, x:x + w])
                    i += 1
                j += 1
            if j == nimgs * 5:
                break
            cv2.imshow('Adding new User', frame)
            if cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        print('Training Model')
        train_model()
        names, rolls, times, l = extract_attendance()
        return render_template('newuser.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)
    else:
        names, rolls, times, l = extract_attendance()
        return render_template('newuser.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/attendance')
def attendance():
    names, rolls, times, l = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('attendance.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

    ret = True
    video_capture = VideoStream(src=0).start()
    eyes_detected = defaultdict(str)

    while ret and app_running:
        frame = video_capture.read()

        frame = detect_and_display(frame, names, eyes_detected)

        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27 or not app_running:
            app_running = False
            break

    video_capture.stop()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('attendance.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

def detect_and_display(frame, names, eyes_detected):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x,y,w,h) in faces:
        encoding = face_recognition.face_encodings(rgb, [(y, x+w, y+h, x)])[0]

        matches = face_recognition.compare_faces(data["encodings"], encoding)

        name = "Unknown"

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

        face = frame[y:y+h,x:x+w]
        gray_face = gray[y:y+h,x:x+w]

        eyes = []
        
        open_eyes_glasses = open_eyes_detector.detectMultiScale(
            gray_face,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(open_eyes_glasses) == 2:
            eyes_detected[name]+='1'
            for (ex,ey,ew,eh) in open_eyes_glasses:
                cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        else:
            left_face = frame[y:y+h, x+int(w/2):x+w]
            left_face_gray = gray[y:y+h, x+int(w/2):x+w]

            right_face = frame[y:y+h, x:x+int(w/2)]
            right_face_gray = gray[y:y+h, x:x+int(w/2)]

            left_eye = left_eye_detector.detectMultiScale(
                left_face_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags = cv2.CASCADE_SCALE_IMAGE
            )

            right_eye = right_eye_detector.detectMultiScale(
                right_face_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags = cv2.CASCADE_SCALE_IMAGE
            )

            eye_status = '1' 
            for (ex,ey,ew,eh) in right_eye:
                color = (0,255,0)
                pred = predict(right_face[ey:ey+eh,ex:ex+ew], model)
                if pred == 'closed':
                    eye_status='0'
                    color = (0,0,255)
                cv2.rectangle(right_face,(ex,ey),(ex+ew,ey+eh),color,2)
            for (ex,ey,ew,eh) in left_eye:
                color = (0,255,0)
                pred = predict(left_face[ey:ey+eh,ex:ex+ew], model)
                if pred == 'closed':
                    eye_status='0'
                    color = (0,0,255)
                cv2.rectangle(left_face,(ex,ey),(ex+ew,ey+eh),color,2)
            eyes_detected[name] += eye_status

        if isBlinking(eyes_detected[name], 3):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            y = y - 15 if y - 15 > 15 else y + 15
            cv2.putText(frame, 'Real: '+name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
        else:
            if len(eyes_detected[name]) > 20:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                y = y - 15 if y - 15 > 15 else y + 15
                cv2.putText(frame, 'Fake: '+name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 0, 255), 2)

    return frame

def isBlinking(history, maxFrames):
    for i in range(maxFrames):
        pattern = '1' + '0'*(i+1) + '1'
        if pattern in history:
            return True
    return False

def extract_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces

def train_model():
    # Add your training logic here
    pass

def predict(img, model):
    img = Image.fromarray(img, 'RGB').convert('L')
    img = img.resize((IMG_SIZE, IMG_SIZE)).astype('float32')
    img /= 255
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    prediction = model.predict(img)
    if prediction < 0.1:
        prediction = 'closed'
    elif prediction > 0.90:
        prediction = 'open'
    else:
        prediction = 'idk'
    return prediction


if __name__ == '__main__':
    app.run(debug=True)

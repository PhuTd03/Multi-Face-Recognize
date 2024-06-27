from scipy.spatial import distance
from flask import Flask, render_template, Response
import cv2
import numpy as np
from model.model import predict_pipeline

app = Flask(__name__)

font = cv2.FONT_HERSHEY_DUPLEX
color_identify = (255, 255, 255)
color_unknown = (255, 255, 255)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(frame):
    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces


def gen_frame():
    camera = cv2.VideoCapture(0)
    print(f'Camera running status: {camera.isOpened()}')
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            faces = detect_faces(frame)

            for (x, y, w, h) in faces:
                face_roi = frame[y:y + h, x:x + w]
                predicted_name, confidence = predict_pipeline(face_roi)

                if predicted_name == "Waiting":
                    cv2.putText(frame, f'{predicted_name} - {np.round(confidence, 2)}', (x, y - 10), font, 1.0,
                                color_unknown, 1)
                else:
                    cv2.putText(frame, f'{predicted_name} - {np.round(confidence, 2)}', (x, y - 10), font, 1.0,
                                color_identify, 1)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/video_feed')
def video_feed():
    return Response(gen_frame(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, render_template, Response
import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from threading import Thread

app = Flask(__name__)
model = load_model("emotion_model.keras")
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.grabbed, self.frame = self.video.read()
        self.stopped = False
        Thread(target=self.update, args=()).start()

    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.video.read()

    def read(self):
        return self.frame

    def release(self):
        self.stopped = True
        self.video.release()

camera = VideoCamera()

def gen_frames():
    while True:
        frame = camera.read()
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 6)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi = roi_gray.astype("float") / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)
            preds = model.predict(roi)[0]
            max_prob = np.max(preds)
            if max_prob > 0.5:
                label = class_labels[np.argmax(preds)]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
                cv2.putText(frame, label, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run()
    finally:
        camera.release()
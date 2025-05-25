import os
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__, template_folder='templates')

model_path = os.path.join(os.path.dirname(__file__), 'emotion_model.keras')
model = load_model(model_path)

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img.astype('float32') / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    face_img = np.expand_dims(face_img, axis=-1)
    return face_img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    file = request.files['frame']
    img = Image.open(file.stream).convert('RGB')
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return jsonify({'emotion': 'No face detected'})

    (x, y, w, h) = faces[0]
    face = gray[y:y+h, x:x+w]
    face_proc = preprocess_face(face)
    preds = model.predict(face_proc)[0]
    label = class_labels[np.argmax(preds)]

    return jsonify({'emotion': label})

if __name__ == '__main__':
    port = 10000
    app.run(host='0.0.0.0', port=port)

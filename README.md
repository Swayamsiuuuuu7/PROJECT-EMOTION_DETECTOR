# 🎭 Emotion Detection Web App

A simple real-time Emotion Detection System built using Flask, OpenCV, and TensorFlow/Keras.  
This application detects human emotions from facial expressions using a trained deep learning model.

---

## 🚀 Quick Start

bash git clone <your-repo-link> cd project pip install -r requirements.txt python app.py 

Open in browser:
http://localhost:10000

---

## 🚀 Features

- 📷 Detect emotions from uploaded images (camera frames)
- 🧠 Deep learning model for facial emotion recognition
- ⚡ Fast and lightweight Flask backend
- 🎯 Detects 7 emotions:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Sad
  - Surprise
  - Neutral

---

## 🛠️ Tech Stack

- Python
- Flask
- OpenCV
- TensorFlow / Keras
- NumPy
- PIL (Pillow)

---

## 📁 Project Structure

project/ │ ├── app.py ├── emotion_model.keras ├── templates/ │   └── index.html

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
bash git clone <your-repo-link> cd project 

### 2️⃣ Install dependencies
bash pip install -r requirements.txt 

### 3️⃣ Run the application
bash python app.py 

### 4️⃣ Open in browser
http://localhost:10000

---

## 📸 How It Works

1. User uploads an image/frame  
2. Image is converted to grayscale  
3. Face is detected using Haar Cascade  
4. Face is resized to 48x48  
5. Model predicts emotion  
6. Result is returned as JSON  

---

## 🔌 API Endpoint

### POST /predict_emotion

Input:
- Image file (frame)

Output:
json {   "emotion": "Happy" } 

---

## ⚠️ Notes

- Only detects the first face in the image  
- Returns "No face detected" if no face is found  
- Make sure emotion_model.keras is in the same directory  

---

## 📌 Future Improvements

- Real-time webcam streaming  
- Multi-face detection  
- Better UI/UX  
- Emotion confidence scores  
- Deploy on cloud  

---

## 🤝 Contributing

Contributions are welcome!

Steps:
1. Fork the repository  
2. Create a new branch  
3. Make your changes  
4. Submit a Pull Request  

---

## 📜 License

This project is licensed under the **MIT

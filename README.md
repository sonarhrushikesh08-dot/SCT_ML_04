✋ Hand Gesture Recognition System

A real-time Hand Gesture Recognition System built using Deep Learning (CNN)
and Computer Vision to enable intuitive human-computer interaction through gesture-based controls.

📌 Overview
This project uses a Convolutional Neural Network (CNN) trained on a custom gesture 
dataset to classify hand gestures from images and live webcam feed. It can be used 
for gesture-based control systems like media control, navigation, and automation.

🚀 Features
- Real-time hand gesture detection using webcam
- CNN-based image classification model
- Supports multiple gesture classes (10 gestures)
- Smooth prediction using frame buffering (deque)
- Gesture-to-action mapping for system control
- High accuracy with a deep learning model

🧠 Tech Stack
- Python 🐍
- OpenCV for image processing:contentReference[oaicite:0]{index=0}  
- TensorFlow / Keras for deep learning:contentReference[oaicite:1]{index=1}  
- NumPy for numerical computation  
- scikit-learn for dataset splitting  

📂 Dataset
The model is trained on a custom dataset containing the following gestures:
- palm  
- l  
- fist  
- fist_moved  
- thumb  
- index  
- ok  
- palm_moved  
- c  
- down  
Dataset images are resized to 64x64 pixels before training.

| Gesture     | Action        |
| ----------- | ------------- |
| Palm        | Stop / Space  |
| Fist        | Exit / Escape |
| L           | Next / Tab    |
| OK          | Click         |
| Thumbs Up   | Volume Up     |
| Thumbs Down | Volume Down   |

⚙️ Installation
bash
Clone repository
git clone https://github.com/your-username/gesture-recognition.git
cd gesture-recognition
Install dependencies
pip install -r requirements.txt

👨‍💻 Author
- Hrushikesh Sonar  
- Internship Project: Hand Gesture Recognition System  
 

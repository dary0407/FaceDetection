# Face Detection & Emotion-Based AI Chat  
**Author: Paval Daria-Andreea**

This project is a real-time facial emotion recognition application integrated with an intelligent AI chat assistant.  
It uses **MediaPipe Face Landmarker** to detect facial expressions and **Groq's LLaMA 3.3 model** to generate emotion-aware responses.

The system combines:
- Live face detection  
- Emotion classification  
- Emotion-adaptive AI chat  
- Unified interface (camera + chat)  
- Real-time processing  

---

## 1. Features

### 1.1 Emotion Recognition  
Detects five primary emotions in real-time:
- HAPPY  
- SAD  
- ANGRY  
- SURPRISED  
- NEUTRAL  

### 1.2 AI Chat Assistant  
- Uses **Groq API (LLaMA 3.3-70B Versatile)**  
- Generates responses based on the detected emotion  

### 1.3 Unified Interface  
- Left: Live webcam feed  
- Right: Chat interface  

### 1.4 Real-Time Webcam Processing  
- Powered by OpenCV  
- Embedded directly inside Tkinter  

---

## 2. Screenshots

### 2.1 Emotion Recognition Examples

#### Neutral  
<img src="assets/neutral.png" width="500"/>

#### Surprised  
<img src="assets/surprised.png" width="500"/>

#### Happy  
<img src="assets/happy.png" width="500"/>

#### Sad  
<img src="assets/sad.png" width="500"/>

#### Angry  
<img src="assets/angry.png" width="500"/>

---

### 2.2 Full Application (Camera + Chat)
<img src="assets/full_app.png" width="700"/>

---

## 3. How It Works

### 3.1 Camera Capture  
OpenCV continuously captures video frames.

### 3.2 Emotion Detection  
MediaPipe extracts 52 face blendshape values.  
Emotion detection is based on:
- mouth smile  
- frown intensity  
- jaw openness  
- eye widening  

### 3.3 AI Chat  
User text + emotion → Groq API → response.

### 3.4 UI Updating  
Tkinter refreshes camera, emotion, and chat at ~30 FPS.

---

## 4. Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python 3.10 | Core language |
| OpenCV | Webcam capture |
| MediaPipe Face Landmarker | Face analysis |
| Groq API | AI chat responses |
| Tkinter | GUI |
| Pillow | Convert frames to Tkinter images |

---

# 5. Installation & Full Setup Guide

Follow these steps to install and run the project.

---

## 5.1 Download or Clone the Repository

Clone:

bash
git clone https://github.com/YOUR_USERNAME/your_repo_name.git
cd your_repo_name

## 5.2 Check version
Must be 3.10
## 5.3 Create a virtual environment
python -m venv emo

##5.4 Activate the environment
emo\Scripts\Activate.ps1
##5.5 Install dependencies
pip install -r requirements.txt
##5.6 Add Your GROQ API Key
Create a .env and add this line GROQ_API_KEY=your_key_here
##5.7 Ensure the Model File Exists
The file face_landmarker.task must be in the project.
##5.9 Run the application
python main.py

# AUTHOR
Paval Daria-Andreea



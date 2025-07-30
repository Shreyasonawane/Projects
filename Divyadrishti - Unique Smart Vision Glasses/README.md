# Divyadrishti – Smart Vision Glasses

An AI-powered wearable device to assist visually impaired users by detecting objects, obstacles, known faces, and reporting GPS location, with web interface for visualization.

## Features
- Object detection with YOLOv8 and audio feedback
- Obstacle detection with ultrasonic sensors
- Face recognition for familiar people
- GPS location tracking
- Streamlit web interface

## Hardware Used
- Raspberry Pi
- Ultrasonic Sensor (HC-SR04)
- USB Webcam
- GPS Module
- Speaker (for TTS)

## Setup
1. Install dependencies:
```
pip install -r requirements.txt
```
2. Run modules:
- `object_detection.py`: Real-time object detection
- `obstacle_detection.py`: Ultrasonic obstacle alert
- `face_recognition_module.py`: Face detection and alert
- `gps_location.py`: Print location coordinates
- `web_app/app.py`: Launch Streamlit UI

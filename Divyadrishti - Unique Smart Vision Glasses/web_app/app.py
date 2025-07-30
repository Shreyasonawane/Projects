import streamlit as st
import cv2
from PIL import Image
import numpy as np

st.title("Divyadrishti – Unique Smart Vision Glasses Demo")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_cv = np.array(img)
    img_cv = img_cv[:, :, ::-1].copy()

    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    results = model(img_cv)[0]

    for box in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = box
        class_name = model.names[int(class_id)]
        st.write(f"Detected: {class_name}")

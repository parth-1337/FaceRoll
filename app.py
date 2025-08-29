import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("FaceRoll")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image.convert("RGB"))
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)

    face_count = len(faces)
    st.success(f"Number of faces detected: {face_count}")
import cv2
import os
import numpy as np
import streamlit as st
from threading import Thread, Event
import queue
import time

frame_queue = queue.Queue()
stop_event = Event()

def load_registered_faces():
    faces_dir = "faces"
    if not os.path.exists(faces_dir):
        return [], []

    face_images = []
    names = []
    for face_name in os.listdir(faces_dir):
        img_path = os.path.join(faces_dir, face_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            face_images.append(img)
            names.append(os.path.splitext(face_name)[0])

    return face_images, names

def recognize_faces():
    st.title("Face Recognition")

    if 'run' not in st.session_state:
        st.session_state.run = False

    def toggle_webcam():
        st.session_state.run = not st.session_state.run
        if st.session_state.run:
            stop_event.clear()
            Thread(target=process_frames, args=(stop_event,)).start()
        else:
            stop_event.set()

    st.button("Start/Stop Webcam", on_click=toggle_webcam)

    registered_faces, names = load_registered_faces()

    if len(registered_faces) == 0:
        st.error("No registered faces found.")
        return

    frame_window = st.image([])

    def process_frames(stop_event):
        camera_index = 0
        cap = cv2.VideoCapture(camera_index)
        
        # Try multiple indices if the first one fails
        while not cap.isOpened() and camera_index < 5:
            st.warning(f"Warning: Camera at index {camera_index} not accessible. Trying next index...")
            camera_index += 1
            cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            st.error("Error: Could not open any webcam. Please check the connection and try again.")
            return

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Failed to capture frame from webcam.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            frame_window.image(frame)

        cap.release()

    if not st.session_state.run:
        st.info("Webcam is not running.")

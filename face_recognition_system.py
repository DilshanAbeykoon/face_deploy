import cv2
import os
import numpy as np
import streamlit as st
import dlib
from threading import Thread, Event
import queue
import time

# Initialize the queue
frame_queue = queue.Queue()
stop_event = Event()

# Function to load registered faces and their names
def load_registered_faces():
    faces_dir = "faces"
    if not os.path.exists(faces_dir):
        return [], []
    
    face_samples = []
    names = []
    for face_name in os.listdir(faces_dir):
        img_path = os.path.join(faces_dir, face_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            face_samples.append(np.array(img, 'uint8'))
            names.append(os.path.splitext(face_name)[0])

    return face_samples, names

def recognize_faces():
    st.title("Face Recognition")

    # Initialize session state if not already done
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

    face_samples, names = load_registered_faces()

    if len(face_samples) == 0:
        st.error("No registered faces found.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(face_samples, np.array(range(len(names))))

    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    frame_window = st.image([])

    def process_frames(stop_event):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Could not open webcam.")
            return

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Failed to capture frame from webcam.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
                
                # Align the face using dlib's shape predictor
                shape = shape_predictor(gray, face)
                aligned_face = dlib.get_face_chip(frame, shape)

                # Convert the aligned face to grayscale
                aligned_face_gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)

                # Predict using the LBPH recognizer
                face_id, confidence = recognizer.predict(aligned_face_gray)

                if confidence < 100:
                    name = names[face_id]
                    confidence_text = f"  {round(100 - confidence)}%"
                else:
                    name = "Unknown"
                    confidence_text = f"  {round(100 - confidence)}%"

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, name + confidence_text, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            frame_queue.put(frame)
            time.sleep(0.1)

        cap.release()

    if st.session_state.run:
        Thread(target=process_frames, args=(stop_event,)).start()

    while st.session_state.run:
        if not frame_queue.empty():
            frame = frame_queue.get()
            frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


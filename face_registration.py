import cv2
import os
import streamlit as st

def register_face():
    st.title("Face Registration")

    name = st.text_input("Enter your name")
    capture_button = st.button("Capture Face")

    if capture_button and name:
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
        
        st.text("Press 'q' to capture the face.")

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Failed to capture frame from webcam.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow("Capture Face", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    face_img = frame[y:y+h, x:x+w]
                    faces_dir = "faces"
                    if not os.path.exists(faces_dir):
                        os.makedirs(faces_dir)
                    img_name = f"{name}.jpg"
                    local_path = os.path.join(faces_dir, img_name)
                    cv2.imwrite(local_path, face_img)
                    st.success(f"Face captured and saved locally for {name} at {local_path}")
                break

        cap.release()
        cv2.destroyAllWindows()

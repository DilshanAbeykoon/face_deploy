import cv2
import os
import streamlit as st

# Streamlit UI for face registration
def register_face():
    st.title("Face Registration")

    name = st.text_input("Enter your name")
    capture_button = st.button("Capture Face")

    if capture_button and name:
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Could not open webcam.")
            return
        
        st.text("Press 'q' to capture the face.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Failed to capture frame from webcam.")
                break
            cv2.imshow("Capture Face", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

        # Save the captured image locally
        img_name = f"{name}.jpg"
        faces_dir = "faces"
        if not os.path.exists(faces_dir):
            os.makedirs(faces_dir)
        local_path = os.path.join(faces_dir, img_name)
        cv2.imwrite(local_path, frame)
        st.success(f"Face captured and saved locally for {name} at {local_path}")

import streamlit as st
from face_registration import register_face
from face_recognition_system import recognize_faces

st.title("Face Attendance System")

# Sidebar navigation
option = st.sidebar.selectbox("Choose an option", ["Register Face", "Recognize Faces"])

if option == "Register Face":
    register_face()
elif option == "Recognize Faces":
    recognize_faces()

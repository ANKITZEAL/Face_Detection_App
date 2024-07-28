import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.face_detection import load_face_detector, detect_faces
from utils.face_recognition import load_face_recognizer, extract_features, recognize_face

# Load models
face_detector = load_face_detector()
face_recognizer = load_face_recognizer()

# Load embeddings and labels
embeddings = np.load('data/embeddings.npy')
labels = np.load('data/labels.npy')

if len(embeddings.shape) == 3:
    embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[2])

st.title("Face Recognition App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Convert the file to an OpenCV image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Detecting faces...")

    faces = detect_faces(image, face_detector)

    if not faces:
        st.write("No faces detected.")
    else:
        for (x, y, x2, y2, conf) in faces:
            face_img = image[y:y2, x:x2]

            if face_img.size != 0:
                input_embedding = extract_features(face_img, face_recognizer)

                if input_embedding is not None:
                    recognized_person, similarity = recognize_face(input_embedding, embeddings, labels)
                    
                    cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)
                    
                    label = f"{recognized_person}, Sim: {similarity:.2f}"
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    
                    text_x = x + (x2 - x) // 2 - text_size[0] // 2
                    text_y = y + y2 - y + 20

                    cv2.putText(image, label, (text_x, text_y), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)
                else:
                    st.write("Unable to extract features from the face.")
            else:
                st.write("Detected face region is invalid.")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption='Processed Image.', use_column_width=True)
# This script creates a simple face recognition application using Streamlit, OpenCV, and pre-trained models for face detection and recognition. Here's a breakdown of what each part of the script does
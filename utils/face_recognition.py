import cv2
import numpy as np

def load_face_recognizer():
    model_path = 'models/openface_nn4.small2.v1.t7'
    recognizer = cv2.dnn.readNetFromTorch(model_path)
    return recognizer

def extract_features(face_image, recognizer):
    if face_image.size == 0:
        return None
    blob = cv2.dnn.blobFromImage(face_image, 1.0 / 255, (96, 96), (0, 0, 0), True, False)
    recognizer.setInput(blob)
    return recognizer.forward()

def recognize_face(embedding, known_embeddings, known_labels):
    embedding = embedding.flatten()
    distances = np.linalg.norm(known_embeddings - embedding, axis=1)
    min_distance_index = np.argmin(distances)
    return known_labels[min_distance_index], 1 - distances[min_distance_index]

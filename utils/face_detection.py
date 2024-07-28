import cv2
import numpy as np

def load_face_detector():
    model_path = 'models/deploy.prototxt'
    weight_path = 'models/res10_300x300_ssd_iter_140000.caffemodel'
    detector = cv2.dnn.readNetFromCaffe(model_path, weight_path)
    return detector

def detect_faces(image, detector):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
    detector.setInput(blob)
    detections = detector.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype(int)
            startX, startY, endX, endY = max(0, startX), max(0, startY), min(w-1, endX), min(h-1, endY)
            if endX > startX and endY > startY:
                faces.append((startX, startY, endX, endY, confidence))
    return faces
# This code provides the implementation of functions to load a face detector model and detect faces in an image.
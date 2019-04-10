import numpy as np
import cv2

class Face_Detector:

    def __init__(self, cascade_location = 'haarcascade_frontalface_default.xml'):
        self.face_cascade = cv2.CascadeClassifier(cascade_location)
    
    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        rois = []
        for (x, y, w, h) in faces:
            rois.append(frame[y : y + h, x : x + w])
        return rois
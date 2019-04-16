import numpy as np
import cv2

class Haar_Face_Detector:

    def __init__(self, cascade_location = 'haarcascade_frontalface_default.xml'):
        self.face_cascade = cv2.CascadeClassifier(cascade_location)
    
    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        face_coordinates = []
        for (x, y, w, h) in faces:
            face_coordinates.append((x, y, w, h))
        return face_coordinates


class DNN_Face_Detector:

    def __init__(self, model_location, weight_location):
        self.model_location = model_location
        self.weight_location = weight_location
    
    def read_net(model_location, weight_location):
        return cv2.dnn.readNetFromCaffe(model_location, weight_location)
    
    def detect(self, frame):
        DNN_Face_Detector.read_net = staticmethod(DNN_Face_Detector.read_net)

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0, (300, 300),
            (103.93, 116.77, 123.68)
        )
        
        net = DNN_Face_Detector.read_net(self.model_location, self.weight_location)
        net.setInput(blob)
        detections = net.forward()

        coordinates = []
        
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3 : 7] * np.array([w, h, w, h])
                start_x, start_y, end_x, end_y = box.astype('int')
                coordinates.append((start_x, start_y, end_y, end_y))
        
        return coordinates
import cv2
import numpy as np
from utils.Face_Detector import *
from utils.Preprocessor import *
from utils.Model_Loader import *

face_detector = Face_Detector('./front_face_cascade.xml')
preprocessor = Preprocessor(96, 96)
model_loader = Model_Loader('./models/model_2.h5')
model_loader.load_model()

video_capture = cv2.VideoCapture(0)

video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    _, frame = video_capture.read()
    
    rois = face_detector.detect(frame)
    
    try:
        x, y, w, h = rois[0]
        roi = frame[y : y + h, x : x + w]
        preprocessed_roi = preprocessor.process(roi)
        
        eye_x, eye_y = model_loader.get_coordinates(
            preprocessed_roi,
            preprocessed_roi.shape[0],
            preprocessed_roi.shape[1]
        )
        print(eye_x + x, eye_y + y)

        cv2.imshow('ROI', preprocessed_roi)
    except Exception as e:
        print(e)
        pass
    
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
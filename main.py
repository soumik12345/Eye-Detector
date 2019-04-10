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
    cv2.imshow('Frame', frame)
    for i, roi in enumerate(rois):
        preprocessed_roi = preprocessor.process(roi)
        cv2.imshow('ROI_' + str(i), preprocessed_roi)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
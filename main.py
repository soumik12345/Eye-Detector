import cv2
import numpy as np
from utils.Face_Detector import *
from utils.Preprocessor import *
from utils.Model_Loader import *

haar_face_detector = Haar_Face_Detector('./models/front_face_cascade.xml')
preprocessor = Preprocessor(96, 96)
model_loader = Model_Loader('./models/model_2.h5')
model_loader.load_model()

video_capture = cv2.VideoCapture(0)

video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    _, frame = video_capture.read()
    
    rois = haar_face_detector.detect(frame)
    
    try:
        x, y, w, h = rois[0]
        roi = frame[y : y + h, x : x + w]
        preprocessed_roi = preprocessor.process(roi)
        
        eye_x, eye_y = model_loader.get_coordinates(
            preprocessed_roi,
            preprocessed_roi.shape[0],
            preprocessed_roi.shape[1]
        )
        
        # cv2.circle(preprocessed_roi, (int(eye_x[0]), int(eye_y[0])), 5, (0, 255, 0), -1)
        # cv2.circle(preprocessed_roi, (int(eye_x[1]), int(eye_y[1])), 5, (0, 255, 0), -1)

        left_eye_pos = (
            x + int((w / 96) * eye_x[0]),
            y + int((h / 96) * eye_y[0])
        )

        right_eye_pos = (
            x + int((w / 96) * eye_x[1]),
            y + int((h / 96) * eye_y[1])
        )

        cv2.circle(frame, left_eye_pos, 5, (0, 255, 0), -1)
        cv2.circle(frame, right_eye_pos, 5, (0, 255, 0), -1)

        cv2.putText(
            frame,
            'Left Eye: ' + str(left_eye_pos),
            (20, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 255), 
            2, cv2.LINE_AA
        )

        cv2.putText(
            frame,
            'Right Eye: ' + str(right_eye_pos),
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 255), 
            2, cv2.LINE_AA
        )
        
    except Exception as e:
        pass
    
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
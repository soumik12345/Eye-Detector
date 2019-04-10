import numpy as np
import cv2

class Preprocessor:

    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def process(self, frame):
        resized_frame = cv2.resize(frame, (self.width, self.height))
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        normalized_frame = gray_frame / 255.0
        return normalized_frame
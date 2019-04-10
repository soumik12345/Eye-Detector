from tensorflow.keras.models import load_model
import numpy as np

class Model_Loader:

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
    
    def load_model(self):
        self.model = load_model(self.model_path)
        self.model.summary()
    
    def get_coordinates(self, X, scale_width, scale_height):
        X = X.reshape(1, scale_width, scale_height, 1)
        predictions = self.model.predict(X)[0]
        x, y = [], []
        for i in range(len(predictions)):
            if i % 2 == 0:
                x.append(predictions[i] * scale_width)
            else:
                y.append(predictions[i] * scale_height)
        return x, y
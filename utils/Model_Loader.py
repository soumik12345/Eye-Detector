from tensorflow.keras.models import load_model

class Model_Loader:

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
    
    def load_model(self):
        self.model = load_model(self.model_path)
        self.model.summary()
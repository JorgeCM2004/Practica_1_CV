from Detector import Detector

class RANSAC(Detector):
    def __init__(self, test_path, models_path):
        super().__init__(test_path, models_path)

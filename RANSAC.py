from Algorithm import Algorithm
import cv2

class RANSAC(Algorithm):
    def __init__(self, test_path = None, models_path = None):
        super().__init__(test_path, models_path)

    def execute(self):
        for image in self.images:
            source_pts, destiny_pts = self.detector_pi.detect(self.template_img, image)
            H, _ = cv2.findHomography(source_pts, destiny_pts, cv2.RANSAC, 5)
            print(H)

RANSAC(None, None).execute()

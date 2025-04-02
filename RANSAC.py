import cv2
import numpy as np
import matplotlib.pyplot as plt
from Algorithm import Algorithm

class RANSAC(Algorithm):
    def __init__(self, test_path = None, models_path = None):
        super().__init__(test_path, models_path)

    def execute(self):
        for image in self.images:
            source_pts, destiny_pts = self.detector_pi.detect(self.template_img, image)
            H_template2image, _ = cv2.findHomography(source_pts, destiny_pts, cv2.RANSAC, 3)
            H_mm2image = H_template2image @ self.H_mm2template
            H_star = np.linalg.inv(self.K) @ H_mm2image
            lambda_value = np.linalg.norm(H_star[:, 0])
            r1 = H_star[:, 0] / lambda_value
            r2 = H_star[:, 1] / lambda_value
            r3 = np.cross(r1, r2)
            R = np.column_stack((r1, r2, r3))
            t = H_star[:, 2] / lambda_value
            P = self.K @ np.hstack((R, t.reshape(-1, 1)))
            origin_point = P @ np.array([0, 0, 0, 1])
            origin_point_image = (round(origin_point[0] / origin_point[2]), round(origin_point[1] / origin_point[2])) # Importante tupla de enteros.
            AXIS_LENGTH = 30
            points_colors = [(P @ np.array([AXIS_LENGTH, 0, 0, 1]), (255, 0, 0)), (P @ np.array([0, AXIS_LENGTH, 0, 1]), (0, 255, 0)), (P @ np.array([0, 0, AXIS_LENGTH, 1]), (0, 0, 255))] # Puntos: X (Rojo), Y (Verde), Z(Azul).
            image_copy = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
            for point, color in points_colors:
                point_image = (round(point[0] / point[2]), round(point[1] / point[2]))
                cv2.line(image_copy, origin_point_image, point_image, color, 5)


RANSAC(None, None).execute()

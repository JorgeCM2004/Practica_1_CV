import cv2
import numpy as np
from Algorithm import Algorithm
from typing import Literal
from Detector_PI import Detector_PI

class RANSAC(Algorithm):
    def __init__(self, test_path = None,
                 models_path = None,
                 detector: Literal["SIFT", "ORB"] = "SIFT",
                 matcher: Literal["BF", "FLANN"] = "BF"):
        super().__init__(test_path, models_path)

        self.detector_pi = Detector_PI(detector, matcher)

    def execute(self):
        for image in self.images:
            # Obtenemos los puntos de interes de una imagen por medio de la imagen_template
            source_pts, destiny_pts = self.detector_pi.detect(self.template_img, image)
            for i in range(1, 11):
                if len(source_pts) < 100:
                    source_pts, destiny_pts = self.detector_pi.detect(self.template_img, image, 0.8 + i / 10)
                else: break
            # Homografia con los puntos de interes y RANSAC
            H_template2image, _ = cv2.findHomography(source_pts, destiny_pts, cv2.RANSAC, 3)

            H_mm2image = H_template2image @ self.H_mm2template #
            H_star = np.linalg.inv(self.K) @ H_mm2image

            lambda_value = np.linalg.norm(H_star[:, 0])
            r1 = H_star[:, 0] / lambda_value #
            r2 = H_star[:, 1] / lambda_value
            r3 = np.cross(r1, r2)

            R = np.column_stack((r1, r2, r3))
            t = H_star[:, 2] / lambda_value
            P = self.K @ np.hstack((R, t.reshape(-1, 1)))

            origin_point = P @ np.array([0, 0, 0, 1])
            origin_point_image = (round(origin_point[0] / origin_point[2]), round(origin_point[1] / origin_point[2])) # Importante tupla de enteros.
            AXIS_LENGTH = 30
            points_colors = [(P @ np.array([AXIS_LENGTH, 0, 0, 1]), (255, 0, 0)), # X (Rojo)
                            (P @ np.array([0, AXIS_LENGTH, 0, 1]), (0, 255, 0)),  # Y (Verde)
                            (P @ np.array([0, 0, AXIS_LENGTH, 1]), (0, 0, 255))]  # Z (Azul)

            image_copy = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
            for point, color in points_colors:
                point_image = (round(point[0] / point[2]), round(point[1] / point[2]))
                cv2.line(image_copy, origin_point_image, point_image, color, 5)
            self._read_3D_object()
            self.model3d.scale(43.0 / 2)
            self.model3d.translate(np.array([2 * 9 + 10 + 1.5 * 43, 9 + 10 + 2.5 * 43, 43 / 2]).reshape(1, -1))
            self.model3d.plot_on_image(image_copy, P)
            cv2.imshow("3D info on images", cv2.resize(cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR), None, fx=0.3, fy=0.3))
            cv2.waitKey(1000)

RANSAC().execute()
#r"C:\Users\harry\Documents\Proyectos-URJC\Python\Vision_Artificial\Practica_1_CV\imgs_template_real\test"

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

    def execute(self, verbose: bool = False) -> list[tuple[str, list, list]]:
        result = []
        for name, image in zip(self.images_names, self.images):
            # Obtenemos los puntos de interes de una imagen por medio de la imagen_template
            source_pts, destiny_pts = self.detector_pi.detect(self.template_img, image)
            for i in range(1, 11):
                if len(source_pts) < 100:
                    source_pts, destiny_pts = self.detector_pi.detect(self.template_img, image, 0.8 + i / 10)
                else: break
            # Homografia con los puntos de interes y RANSAC
            H_template2image, _ = cv2.findHomography(source_pts, destiny_pts, cv2.RANSAC, 3)

            P = self._calculate_P(H_template2image)

            result.append((name, self._plot_axis_cube_image(image, P, verbose), P.copy()))

        return result

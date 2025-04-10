import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import Literal

class Detector_PI:
    def __init__(self,
                 descriptor_name: Literal["ORB", "SIFT"] = "ORB",
                 matcher_name: Literal["BF", "FLANN"] = "BF"):
        self.change_descriptor(descriptor_name)
        self.change_matcher(matcher_name)

    def change_descriptor(self, descriptor_name: Literal["ORB", "SIFT"] = "ORB"):
        self.descriptor_name = descriptor_name
        self._instance_descriptor()

    def change_matcher(self, matcher_name: Literal["BF", "FLANN"] = "BF"):
        self.matcher_name = matcher_name
        self._instance_matcher()

    def detect(self, source_image, destiny_image, threshold: int = 0.8, verbose: bool = False):
        # Copiado de las diapositivas. (¿Cambiar?)
        kp1, des1 = self.descriptor.detectAndCompute(source_image,None)
        kp2, des2 = self.descriptor.detectAndCompute(destiny_image,None)
        matches = self.matcher.knnMatch(des1, des2, 2)
        good = [m for m, n in matches if m.distance < threshold * n.distance]
        if verbose:
            img3 = cv2.drawMatchesKnn(source_image, kp1, destiny_image, kp2, [[m] for m in good], None, flags=2)
            plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
            plt.show()
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        return src_pts, dst_pts


    def _instance_descriptor(self):
        match self.descriptor_name:
            case "ORB":
                self.descriptor = cv2.ORB_create()
            case "SIFT":
                self.descriptor = cv2.SIFT_create()
            case _:
                raise ValueError(f"El descriptor {self.descriptor_name} no es válido.")

    def _instance_matcher(self):
        match self.matcher_name:
            case "BF":
                self.matcher = cv2.BFMatcher()
            case "FLANN":
                self.matcher = cv2.FlannBasedMatcher()
            case _:
                raise ValueError(f"El matcher {self.matcher_name} no es válido.")

import cv2
import numpy as np
from copy import copy
from Algorithm import Algorithm

class Detector(Algorithm):
    def __init__(self):
        """
        Inicializa el detector con la imagen template.
        :param template_path: Ruta de la imagen template.
        """
        self.template = self.template_img
        if self.template is None:
            raise ValueError("No se pudo cargar la imagen template.")
        self.detector = cv2.SIFT_create()  # Usamos SIFT para detectar características
        self.template_keypoints, self.template_descriptors = self.detector.detectAndCompute(self.template, None)

    def detect_and_compute_homography(self):
        """
        Detecta objetos dentro del marco rojo y calcula la matriz de homografía.
        :param image_path: Ruta de la imagen de entrada.
        :return: Matriz de homografía (3x3) o None si no se encuentra suficiente correspondencia.
        """
        # Cargar la imagen de entrada
        image = self.images[0]
        if image is None:
            raise ValueError("No se pudo cargar la imagen de entrada.")

        # Detectar características en la imagen de entrada
        keypoints, descriptors = self.detector.detectAndCompute(image, None)

        # Coincidencia de características entre la imagen y el template
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(self.template_descriptors, descriptors)
        matches = sorted(matches, key=lambda x: x.distance)

        # Verificar si hay suficientes coincidencias
        if len(matches) < 4:
            print("No se encontraron suficientes coincidencias.")
            return None

        # Extraer puntos clave
        src_pts = np.float32([self.template_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Calcular la matriz de homografía
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H

    def draw_detected_area(self, H):
        """
        Dibuja el área detectada en la imagen de entrada usando la matriz de homografía.
        :param image_path: Ruta de la imagen de entrada.
        :param H: Matriz de homografía.
        :return: Imagen con el área detectada dibujada.
        """
        image = self.images[0]
        if image is None:
            raise ValueError("No se pudo cargar la imagen de entrada.")

        # Obtener las dimensiones del template
        h, w = self.template.shape

        # Definir los puntos del marco del template
        template_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

        # Transformar los puntos del template a la imagen de entrada
        detected_corners = cv2.perspectiveTransform(template_corners, H)

        # Dibujar el marco en la imagen
        detected_image = cv2.polylines(image, [np.int32(detected_corners)], True, (0, 0, 255), 3, cv2.LINE_AA)
        return detected_image
    
# Crear una instancia del detector
detector = Detector()

# Detectar y calcular la homografía
H = detector.detect_and_compute_homography()

if H is not None:
    # Dibujar el área detectada
    result_image = detector.draw_detected_area(H)
    cv2.imshow("Resultado", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No se pudo calcular la homografía.")
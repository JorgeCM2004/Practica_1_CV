import os
import numpy as np
import cv2
from glob import glob
from Detector_PI import Detector_PI

class Algorithm:
    def __init__(self, test_path: str, models_path: str):
        self.dir_path = os.path.dirname(__file__) # Ruta absoluta donde se encuentra la carpeta madre.

        self.test_path = test_path

        self.models_path = models_path

        self.detector_pi = Detector_PI("SIFT")

        # Leer y almacenar variables y parametros dados por el problema.
        self._read_all()

        # Calcular la matriz que transforma de milimetros a pixeles.
        self._calculate_H_mm2template()

    def _read_all(self):
        self._read_images()
        self._read_3D_object()
        self._read_template_cropped()
        self._read_intrinsct()

    def _read_images(self):
        self.images = []
        # Imágenes si se recibe la ruta.
        if self.test_path:
            path_list = sorted(glob(os.path.join(self.test_path, "*.jpg")))
        # Imágenes si hay que buscar la carpeta.
        else:
            path_list = sorted(glob(os.path.join(self.dir_path, "imgs_template_real", "secuencia", "*.jpg")))

        for file in path_list:
            aux = cv2.imread(file)
            self.images.append(aux)
            # Revisar si la ultima imagen se leyó correctamente.
            if self.images[-1] is None:
                raise ValueError(f"La imagen {file} no se pudo cargar correctamente.")


    def _read_3D_object(self):
        pass

    def _read_template_cropped(self):
        try:
            # Intrinsics si se recibe la ruta.
            if self.test_path:
                self.template_img = cv2.imread(os.path.join(self.test_path, "template_cropped.png"))
            # Intrinsics si hay que buscar la carpeta.
            else:
                self.template_img = cv2.imread(os.path.join(self.dir_path, "imgs_template_real", "secuencia", "template_cropped.png")) # Si no se especifica se usa la carpeta "secuencia" como prederterminada.
        except:
            raise FileNotFoundError("No se encuentra 'template_cropped.png' en la ruta especificada.")

        if self.template_img is None:
            raise ValueError("La imagen 'template_cropped.png' no se pudo cargar correctamente.")

    def _read_intrinsct(self):
        try:
            # Intrinsics si se recibe la ruta.
            if self.test_path:
                self.K = np.loadtxt(os.path.join(self.test_path, "intrinsics.txt"))
            # Intrinsics si hay que buscar la carpeta.
            else:
                self.K = np.loadtxt(os.path.join(self.dir_path, "imgs_template_real", "secuencia", "intrinsics.txt"))
        except:
            raise FileNotFoundError("Error al cargar 'intrinsics.txt'.")

    def _calculate_H_mm2template(self):
        # Esquinas: derecha arriba, izquierda arriba, derecha abajo, izquierda abajo.
        mm_mat = np.array([[0, 0], [210, 0], [0, 185], [210, 185]], dtype = np.float32) # Milimetros
        pixel_mat = np.array([[0, 0], [1484, 0], [0, 1307], [1484, 1307]], dtype = np.float32) # Píxeles
        self.H_mm2template = cv2.getPerspectiveTransform(mm_mat, pixel_mat)

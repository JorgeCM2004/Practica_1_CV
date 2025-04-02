import os
import numpy
import cv2
from glob import glob
from Detector_PI import Detector_PI

class Algorithm:
    def __init__(self, test_path: str, models_path: str):
        self.dir_path = os.path.dirname(__file__) # Ruta absoluta donde se encuentra la carpeta madre.

        self.test_path = test_path

        self.models_path = models_path

        self.detector_pi = Detector_PI("SIFT")

        self._read_all()

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
                self.K = numpy.loadtxt(os.path.join(self.test_path, "intrinsics.txt"))
            # Intrinsics si hay que buscar la carpeta.
            else:
                self.K = numpy.loadtxt(os.path.join(self.dir_path, "imgs_template_real", "secuencia", "intrinsics.txt"))
        except:
            raise FileNotFoundError("Error al cargar 'intrinsics.txt'.")


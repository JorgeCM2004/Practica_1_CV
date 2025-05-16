import os
import argparse
from glob import glob
import cv2
from typing import Literal
from Saver import Saver
from RANSAC import RANSAC
from MarcoRojo import MarcoRojo
from DetectorTexturas import Detector_Texturas

class Runner:
    def __init__(self,
                 detector: Literal["KEYPOINTS", "MARCO_ROJO", "DETECTOR_TEXTURAS"] = "KEYPOINTS",
                 test_path: str = None,
                 models_path: str = None,
                 result_path: str = None):

        # Definir nombre del detector:
        self.detector_name = detector
        print(f"Detector seleccionado: '{self.detector_name}'.")

        # Definir path de la carpeta de imagenes:
        self.test_path = test_path

        # Definir path de la carpeta de los modelos 3D:
        self.models_path = models_path

        # Definir path de la carpeta donde guardar los resultados:
        self.result_path = result_path

        # Crear detector.
        self._instance_detector()

    def _instance_detector(self):
        match self.detector_name:
            case "KEYPOINTS":
                self.detector = RANSAC(self.test_path, self.models_path)
            case "MARCO_ROJO":
                self.detector = MarcoRojo(self.test_path, self.models_path)

            case "DETECTOR_TEXTURAS":
                self.detector = Detector_Texturas(self.test_path, self.models_path)
            case _:
                raise ValueError("El nombre del detector no existe.")

    def run(self, save: bool = True, verbose: bool = False):
        # Comprobación de tipos en parametro save.
        if not isinstance(save, bool):
            raise TypeError("El tipo de 'save' debe ser booleano (True o False).")

        # Comprobación de tipos en parametro verbose.
        if not isinstance(verbose, bool):
            raise TypeError("El tipo de 'verbose' debe ser booleano (True o False).")

        l_name_image_Pmatrix = self.detector.execute(verbose)

        if save:
            # Crear guardado.
            saver = Saver(self.result_path)
            saver.save_images(l_name_image_Pmatrix)
            saver.save_txt(l_name_image_Pmatrix)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Crea y ejecuta un detector sobre las imágenes de test.')
    parser.add_argument(
        '--detector', default="DETECTOR_TEXTURAS", help = 'Nombre del detector a ejecutar.')
    parser.add_argument(
        '--test_path', default = None, help = 'Carpeta con las imágenes de test.')
    parser.add_argument(
        '--models_path', default = None, help = 'Carpeta con los modelos 3D (.obj).')
    parser.add_argument(
        '--result_path', default = None, help = 'Carpeta donde se guardaran las imagenes procesadas.')
    args = parser.parse_args()

    runner = Runner(args.detector, args.test_path, args.models_path, args.result_path)
    runner.run()

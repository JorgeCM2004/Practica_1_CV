import os
import argparse
from glob import glob
import numpy as np
import cv2
from typing import Literal

class Runner:
    def __init__(self,
                 detector: Literal["KEYPOINTS"] = "KEYPOINTS",
                 test_path: str = None,
                 models_path: str = None):

        # Definir nombre del detector:
        self.detector_name = detector

        # Definir path de la carpeta de imagenes:
        if test_path:
            self.test_path = test_path
        else:
            self.test_path = None

        # Definir path de la carpeta de los modelos 3D:
        if models_path:
            self.models_path = models_path
        else:
            self.models_path = None

    def run(self, save: bool = True):
        # Comprobación de tipos en parametro save.
        if not isinstance(save, bool):
            raise TypeError("El tipo de 'save' debe ser booleano (True o False).")
        pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Crea y ejecuta un detector sobre las imágenes de test')
    parser.add_argument(
        '--detector', type=str, nargs="?", default="KEYPOINTS", help='Nombre del detector a ejecutar')
    parser.add_argument(
        '--test_path', default = None, help = 'Carpeta con las imágenes de test')
    parser.add_argument(
        '--models_path', default = None, help = 'Carpeta con los modelos 3D (.obj)')
    args = parser.parse_args()


    # Definir el tipo de algoritmo de detección a utilizar
    print("Detector seleccionado " + args.detector)
    planar_localizer_name = args.detector

    """    # Cargar la imagen de la plantilla escaneada
    template_img_path = os.path.join(args.test_path, "template_cropped.png")
    template_img = cv2.imread(template_img_path)
    if template_img is None:
        print("No puedo encontrar la imagen " + template_img_path)"""# Hecho dentro de detector.

    """# Leer la matriz de intrínsecos de la cámara.
    K = np.load(os.path.join(args.test_path, "intrinsics.txt"))""" # Hecho dentro de detector.

    # Crear el detector de la plantilla pertinente (con KEYPOINTS u otro).
#    if args.detector == "KEYPOINTS:


    # Cargar el modelo 3D del cubo y colocarlo en el lugar pedido.
    model_3d_file = os.path.join(args.models_path, "cubo.obj")
    print("Cargando el modelo 3D " + model_3d_file)

    # Recorrer las imágenes en el directorio seleccionado y procesarlas.
    print("Probando el detector " + args.detector + " en " + args.test_path)
    paths = sorted(glob(os.path.join(args.test_path, "*.jpg")))
    for f in paths:
        query_img_path = f
        if not os.path.isfile(query_img_path):
            continue

        query_img = cv2.imread(query_img_path)
        if query_img is None:
            print("No puedo encontrar la imagen " + query_img_path)
            continue

        # Localizar el plano en la imagen y calcular R, t -> P


        # Mostrar los ejes del sistema de referencia de la plantilla sobre una copia de la imagen de entrada.
        plot_img = query_img.copy()


        # Mostrar el modelo 3D del cubo sobre la imagen plot_img


        # Mostrar el resultado en pantalla.
        cv2.imshow("3D info on images", cv2.resize(plot_img, None, fx=0.3, fy=0.3))
        cv2.waitKey(1000)

        # Guardar la imagen resultado en el directorio os.path.join(images_path, "resultado_imgs")

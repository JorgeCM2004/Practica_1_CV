import os
import numpy as np
import cv2
from glob import glob
from model3d import Model3D

class Algorithm:
    def __init__(self, test_path: str, models_path: str):
        self.dir_path = os.path.dirname(__file__) # Ruta absoluta donde se encuentra la carpeta madre.

        self.test_path = test_path

        self.models_path = models_path

        # Leer y almacenar variables y parametros dados por el problema.
        self._read_all()

        # Calcular la matriz que transforma de milimetros a pixeles.
        self._calculate_H_mm2template()

    def _plot_axis_cube_image(self, image, P, verbose: bool = False):
        origin_point = P @ np.array([0, 0, 0, 1])
        origin_point_image = (round(origin_point[0] / origin_point[2]), round(origin_point[1] / origin_point[2])) # Importante tupla de enteros.
        AXIS_LENGTH = 30
        points_colors = [(P @ np.array([AXIS_LENGTH, 0, 0, 1]), (255, 0, 0)), # X (Rojo)
                        (P @ np.array([0, AXIS_LENGTH, 0, 1]), (0, 255, 0)),  # Y (Verde)
                        (P @ np.array([0, 0, AXIS_LENGTH, 1]), (0, 0, 255))]  # Z (Azul)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for point, color in points_colors:
            point_image = (round(point[0] / point[2]), round(point[1] / point[2]))
            cv2.line(image, origin_point_image, point_image, color, 5)
        self._read_3D_object()
        self.model3d.scale(43.0 / 2)
        self.model3d.translate(np.array([2 * 9 + 10 + 1.5 * 43, 9 + 10 + 2.5 * 43, 43 / 2]).reshape(1, -1))
        self.model3d.plot_on_image(image, P)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if verbose:
            cv2.imshow("3D info on images", cv2.resize(image, None, fx=0.3, fy=0.3))
            cv2.waitKey(1000)
        return image

    def _calculate_P(self, H_template2image):
        H_mm2image = H_template2image @ self.H_mm2template
        H_star = np.linalg.inv(self.K) @ H_mm2image

        lambda_value = np.linalg.norm(H_star[:, 0])
        r1 = H_star[:, 0] / lambda_value #
        r2 = H_star[:, 1] / lambda_value
        r3 = np.cross(r1, r2)

        R = np.column_stack((r1, r2, r3))
        t = H_star[:, 2] / lambda_value
        P = self.K @ np.hstack((R, t.reshape(-1, 1)))
        return P

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

        self.images_names = [os.path.basename(path) for path in path_list]

        for file in path_list:
            aux = cv2.imread(file)
            self.images.append(aux)
            # Revisar si la ultima imagen se leyó correctamente.
            if self.images[-1] is None:
                raise ValueError(f"La imagen {file} no se pudo cargar correctamente.")


    def _read_3D_object(self):
        self.model3d = Model3D()
        if self.models_path:
            self.model3d.load_from_obj(self.models_path)
        else:
            self.model3d.load_from_obj(os.path.join(self.dir_path, "3d_models", "cubo.obj"))

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
        # Esquinas: Izquierda arriba, derecha arriba, izquierda abajo, derecha abajo.
        mm_mat = np.array([[0, 0], [0, 210], [185, 0], [185, 210]], dtype = np.float32) # Milimetros (vertical, horizontal)
        pixel_mat = np.array([[0, 0], [self.template_img.shape[1], 0], [0, self.template_img.shape[0]], [self.template_img.shape[1],  self.template_img.shape[0]]], dtype = np.float32) # Píxeles (horizontal, vertical)
        self.H_mm2template = cv2.getPerspectiveTransform(mm_mat, pixel_mat)

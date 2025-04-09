import os
import cv2
import shutil
import numpy as np

class Saver:
	def __init__(self, path: str = None):
		if path is None:
			self.dir_path = os.path.join(os.path.dirname(__file__), "resultado_imgs")
		else:
			self.dir_path = path
		if os.path.exists(self.dir_path):
			shutil.rmtree(self.dir_path)
		os.mkdir(self.dir_path)

	def save_images(self, names_images_Ps: list) -> bool:
		for name, image, _ in names_images_Ps:
			cv2.imwrite(os.path.join(self.dir_path, name), image)
		return True

	def save_txt(self, names_images_Ps: list) -> bool:
		with open(os.path.join(self.dir_path, "resultado.txt"), "w") as file:
			file.write("<nombre_fichero_imagen>; <x0>; <y0>; <x1>; <y1>; <x2>; <y2>; <x3>; <y3>\n")
			for name, _, P in names_images_Ps:
				points = np.array([[0, 0, 0, 1], [0, 210, 0, 1], [185, 210, 0, 1], [185, 0, 0, 1]], dtype = np.float32)
				linea = (
					f"{name}; "
					f"{(P @ points[0])[0] / (P @ points[0])[2]}; "
					f"{(P @ points[0])[1] / (P @ points[0])[2]}; "
					f"{(P @ points[1])[0] / (P @ points[1])[2]}; "
					f"{(P @ points[1])[1] / (P @ points[1])[2]}; "
					f"{(P @ points[2])[0] / (P @ points[2])[2]}; "
					f"{(P @ points[2])[1] / (P @ points[2])[2]}; "
					f"{(P @ points[3])[0] / (P @ points[3])[2]}; "
					f"{(P @ points[3])[1] / (P @ points[3])[2]}\n"
				)
				file.write(linea)
		return True

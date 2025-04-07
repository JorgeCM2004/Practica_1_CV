import os
class Saver:
	def __init__(self, path: str = None):
		if path is None:
			self.dir_path = os.path.join(os.path.dirname(__file__), "resultado_imgs")
		else:
			self.dir_path = path
		if os.path.exists(self.dir_path):
			os.removedirs(self.dir_path)

	def save_image(self, image):
		pass

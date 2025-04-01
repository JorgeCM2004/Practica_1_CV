import os
from typing import Literal

class Ransac:
    def __init__(self):
        self.file_path = os.path.abspath(__file__) # Ruta absoluta donde se encuentra el archivo.
        #self.state = None

    def secuencia(self): # Usa archivos de la carpeta secuencia.
        self.state = "secuencia"

    def test(self): # Usa archivos de la carpeta de test.
        self.state = "test"

    def _read_intrinsct(self):
        with open(os.path.join(self.file_path, "imgs_template_real", "secuencia", "intri")) as file:
            pass

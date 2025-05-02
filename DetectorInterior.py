import cv2
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from Algorithm import Algorithm
from itertools import product
from itertools import combinations
from tqdm import tqdm
import math
import itertools


class Detector_Interior(Algorithm):
    def __init__(self, test_path=None, models_path=None):
        super().__init__(test_path, models_path)

    def _marco_grande(self, imagen): # Encuentra el contorno mas grande que cumpla con los requisitos

        imgCanny = cv2.Canny(imagen, 50, 150)
        contornos, jerarquia = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        alto, ancho = imagen.shape[:2]
        marco_grande, _ = self._biggestContour(contornos, alto, ancho)

        return marco_grande

    def _biggestContour(self, contours, alto, ancho): # Encuentra el contorno mas grande con 4 vertices

        biggest = np.array([])
        max_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area > (alto * ancho * 0.05):  # Filtrar contornos pequeños
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area
        return biggest, max_area

    def _umbralizado(self, imagen_original, color="blanco"): # Aplica un umbral basado en el color especificado

        colores = {
            "blanco": {
                "lower": np.array([0, 0, 200]),
                "upper": np.array([180, 50, 255])
            },
            "rojo": {
                "lower1": np.array([0, 100, 100]),
                "upper1": np.array([10, 255, 255]),
                "lower2": np.array([160, 100, 100]),
                "upper2": np.array([180, 255, 255])
            },
            "azul": {
                "lower": np.array([100, 150, 0]),
                "upper": np.array([140, 255, 255])
            }
        }

        if color not in colores:
            raise ValueError("Color ", color, " no soportado. Usa 'blanco', 'rojo' o 'azul'.")

        hsv = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2HSV)

        if color == "blanco":
            lower = colores["blanco"]["lower"]
            upper = colores["blanco"]["upper"]
            mask = cv2.inRange(hsv, lower, upper)
        elif color == "rojo":
            lower1 = colores["rojo"]["lower1"]
            upper1 = colores["rojo"]["upper1"]
            lower2 = colores["rojo"]["lower2"]
            upper2 = colores["rojo"]["upper2"]
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        elif color == "azul":
            lower = colores["azul"]["lower"]
            upper = colores["azul"]["upper"]
            mask = cv2.inRange(hsv, lower, upper)

        resultado = cv2.bitwise_and(imagen_original, imagen_original, mask=mask)
        return resultado

    def _ordenar_esquinas(self, puntos):
        
        # Ordena los puntos en el orden: superior izquierda, superior derecha,
        # inferior derecha, inferior izquierda, asegurando consistencia en la orientación.

        # Asegurarse de que los puntos tienen las proporciones correctas
        puntos = puntos.reshape((4, 2))

        # Calcularmos el centroide de los puntos
        centro = np.mean(puntos, axis=0)

        # Calcular el angulo de cada punto respecto al centroide
        def angulo_desde_centro(punto):
            return np.arctan2(punto[1] - centro[1], punto[0] - centro[0])

        # Ordenar los puntos en sentido antihorario
        puntos_ordenados = sorted(puntos, key=angulo_desde_centro)

        puntos_ordenados = np.array(puntos_ordenados, dtype=np.float32) # Convierte a array pero de numpy

        # Verificar la orientacion usando el producto cruzado
        # Producto cruzado entre los vectores (p0 -> p1) y (p0 -> p3)
        vector1 = puntos_ordenados[1] - puntos_ordenados[0]
        vector2 = puntos_ordenados[3] - puntos_ordenados[0]
        cross_product = np.cross(vector1, vector2)

        # Si el producto cruzado es negativo, invertir el orden para mantener la orientación
        if cross_product < 0:
            puntos_ordenados[[1, 3]] = puntos_ordenados[[3, 1]]

        return puntos_ordenados

    def _ordenar_esquinas_con_patron(self, puntos, patron): # Ordena las esquinas del marco en función de un patrón interno.

        puntos = puntos.reshape((4, 2))

        # Calcular el centroide del marco
        centro = np.mean(puntos, axis=0)

        # Determinar la esquina más cercana al patrón interno
        distancias = [np.linalg.norm(punto - patron[:2]) for punto in puntos]
        esquina_cercana = np.argmin(distancias)

        # Reordenar las esquinas comenzando desde la más cercana al patrón
        puntos_ordenados = np.roll(puntos, -esquina_cercana, axis=0)

        return puntos_ordenados
    
    def completar_cuadrado_con_3_puntos(self, puntos):
        if puntos.shape[2] != 3:
            raise ValueError("Se necesitan al menos 3 puntos para completar el cuadrado.")

        puntos = puntos.reshape((3, 2))

        max_dist = 0
        diagonal = (0, 1)
        for i in range(3):
            for j in range(i + 1, 3):
                dist = np.linalg.norm(puntos[i] - puntos[j])
                if dist > max_dist:
                    max_dist = dist
                    diagonal = (i, j)

        A = puntos[diagonal[0]]
        B = puntos[diagonal[1]]
        C = puntos[3 - diagonal[0] - diagonal[1]] 

        centro = (A + B) / 2

        D = 2 * centro - C

        return np.array([A, B, C, D], dtype=np.float32)

    def detectar_patron_interno(self, imagen, marco): # Detecta un patrón interno dentro del marco rojo.

        # Crear una mascara para el area dentro del marco
        mask = np.zeros(imagen.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [marco.astype(int)], -1, 255, -1)

        # Aplicar la mascara a la imagen 
        imagen_interna = cv2.bitwise_and(imagen, imagen, mask=mask)

        # Convertir a escala de grises y buscar círculos
        gray = cv2.cvtColor(imagen_interna, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2, 
            minDist=20, 
            param1=50, 
            param2=30, 
            minRadius=5, 
            maxRadius=50
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            print("Círculos detectados:", len(circles))
            return circles[0]  # Devuelve el primer círculo detectado 
        else:
            print("No se detectaron patrones internos.")
            return None

    def evaluar_homografia(self, marco_grande, puntosTemplate): # Compara las posibles homografias de los 4 puntos para quedarse con la que esta bien oriantada

        # Genera todas las permutaciones posibles de las esquinas
        permutaciones = list(itertools.permutations(marco_grande, 4))
        mejor_homografia = None
        mejor_puntaje = float('inf')  # Menor puntaje = Menor diferencia con template = es mejor 

        for permutacion in permutaciones:
            puntosImagen = np.array(permutacion, dtype=np.float32)
            H = cv2.getPerspectiveTransform(puntosTemplate, puntosImagen)

            # Evaluar la homografia
            puntaje = self._evaluar_calidad_homografia(H, puntosTemplate, puntosImagen)
            if puntaje < mejor_puntaje:
                mejor_puntaje = puntaje
                mejor_homografia = H

        return mejor_homografia

    def _evaluar_calidad_homografia(self, H, puntosTemplate, puntosImagen): # Calculo del error entre template y homogradia

        # Transformar los puntos de la plantilla usando la homografía
        puntos_transformados = cv2.perspectiveTransform(np.array([puntosTemplate]), H)[0]

        # Calcular el error de reproyección
        error = np.linalg.norm(puntosImagen - puntos_transformados, axis=1)
        return np.sum(error)  # Suma de los errores

    def mostrar_imagen(self, imagen):

        plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
        plt.show()

    def execute(self):

        if not self.images:
            print("¡¡ No hay imágenes para procesar.")
            return

        imagen = self.images[12].copy()
        imagen = cv2.bilateralFilter(imagen, d=9, sigmaColor=75, sigmaSpace=75)

        # self.mostrar_imagen(imagen)
        try:
            # Mejora imagen
            clashe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            imagen_mejorada = clashe.apply(gray)
            # self.mostrar_imagen(imagen_mejorada)

            canal_rojo = imagen[:,:,2]
            imagen_mejorada = cv2.Canny(canal_rojo, 50, 150)

            # Umbralizar la imagen
            img_umbralizada = self._umbralizado(imagen, "rojo")
            

            # Encontrar el marco más grande
            marco_grande = self._marco_grande(img_umbralizada)

            if marco_grande.shape[0] == 3:
                marco_grande = self.completar_cuadrado_con_3_puntos(marco_grande)

            if marco_grande.shape[0] == 0:
                print("NINGUN Punto detectado")
                plt.imshow(img_umbralizada)
                plt.show()

            elif marco_grande.shape[0] != 4:
                print(f"¡¡ No se han podido obtener los 4 puntos del marco.")
                return

            # Detectar patrón interno
            patron_interno = self.detectar_patron_interno(imagen, marco_grande)

            if patron_interno is not None:
                marco_grande = self._ordenar_esquinas_con_patron(marco_grande, patron_interno)
            else:
                marco_grande = self._ordenar_esquinas(marco_grande)

            # Dibujar el contorno en la imagen original
            COLOR_AZUL = (255, 0, 0)
            cv2.drawContours(imagen, [marco_grande.astype(int)], -1, COLOR_AZUL, 2)

            # Obtener el ancho y alto de la imagen template
            width = self.template_img.shape[1]  # Ancho de la imagen template
            height = self.template_img.shape[0]  # Alto de la imagen template
            puntosTemplate = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

            # Usar los puntos del contorno detectado como puntos de la imagen
            puntosImagen = self._ordenar_esquinas(marco_grande).astype(np.float32)
            H_template2image = cv2.getPerspectiveTransform(puntosTemplate, puntosImagen)
 
            P = self._calculate_P(H_template2image)

            self._plot_axis_cube_image(imagen, P, True)

        except Exception as e:
            print("¡¡ Ocurrió un error durante el procesamiento: ", e)

        
Detector_Interior(None, None).execute()
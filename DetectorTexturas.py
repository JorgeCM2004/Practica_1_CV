import cv2
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from Algorithm import Algorithm
from itertools import product
from itertools import combinations
from tqdm import tqdm



class Detector_Texturas(Algorithm):
    def __init__(self, test_path=None, models_path=None):
        super().__init__(test_path, models_path)

    def _homografia_fija(self):
        # Genera una matriz de homografía identidad fija 
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

    # -------------------------------
    # 1. Métodos de Preprocesamiento
    # -------------------------------

    def _umbralizado(self, imagen_original, ajuste_factor=0, filtro_tamano=5):

        if imagen_original is None or len(imagen_original.shape) != 3:
            # La imagen proporcionada no es válida.
            raise ValueError("")

        # Convertir la imagen a espacio de color HSV
        hsv = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2HSV)

        # Definir los umbrales iniciales para el color rojo
        lower1 = np.array([0, 160, 120])
        upper1 = np.array([60, 255, 255])
        lower2 = np.array([160, 90, 100])
        upper2 = np.array([180, 255, 255])

        # Ajustar dinámicamente los valores de umbral
        mean_val = np.mean(hsv[:, :, 0])  # Promedio del canal H
        lower1[0] = max(0, lower1[0] - int(mean_val * ajuste_factor))
        upper1[0] = min(180, upper1[0] + int(mean_val * ajuste_factor))
        lower2[0] = max(0, lower2[0] - int(mean_val * ajuste_factor))
        upper2[0] = min(180, upper2[0] + int(mean_val * ajuste_factor))

        # Crear máscaras para los dos rangos de rojo
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)

        # Combinar las máscaras
        mask = cv2.bitwise_or(mask1, mask2)

        # Aplicar un filtro adaptativo para suavizar la máscara
        mask = cv2.medianBlur(mask, filtro_tamano)

        # Aplicar la máscara a la imagen original
        resultado = cv2.bitwise_and(imagen_original, imagen_original, mask=mask)
        return resultado
    
    # --------------------------------------------------
    # 2. Deteccion de contornos y Métodos de Ordenación
    # --------------------------------------------------

    def _marco_grande(self, imagen): # Encuentra el contorno mas grande que cumpla con los requisitos

        blurred = cv2.GaussianBlur(imagen, (5, 5), 0)
        imgCanny = cv2.Canny(blurred, 50, 150)
        contornos, jerarquia = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        alto, ancho = imagen.shape[:2]
        marco_grande, _ = self._biggestContour(contornos, alto, ancho)

        return marco_grande

    def _biggestContour(self, contours, alto, ancho): # Encuentra el contorno mas grande con 4 vertices

        biggest = np.array([])
        max_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area > (alto * ancho * 0.04):  # Filtrar contornos pequeños
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area
        return biggest, max_area

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
        vector1 = puntos_ordenados[3] - puntos_ordenados[0]
        vector2 = puntos_ordenados[1] - puntos_ordenados[0]
        cross_product = np.cross(vector1, vector2)

        # Si el producto cruzado es negativo, invertir el orden para mantener la orientación
        if cross_product < 0:
            puntos_ordenados[[1, 3]] = puntos_ordenados[[3, 1]]
        
        puntos_ordenados = np.array([
                puntos_ordenados[1],  # Inferior izquierda
                puntos_ordenados[0],  # Inferior derecha
                puntos_ordenados[3],  # Superior derecha
                puntos_ordenados[2]   # Superior izquierda
            ], dtype=np.float32)

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

        # Ajustar el orden para garantizar que sea consistente con el formato esperado
        # superior izquierda, superior derecha, inferior derecha, inferior izquierda
        if np.cross(puntos_ordenados[1] - puntos_ordenados[0], puntos_ordenados[3] - puntos_ordenados[0]) < 0:
            puntos_ordenados[[1, 3]] = puntos_ordenados[[3, 1]]  # Intercambiar si está al revés

        puntos_ordenados = np.flip(puntos_ordenados, axis=0)

        puntos_ordenados = np.array([
                puntos_ordenados[1],  # Inferior izquierda
                puntos_ordenados[0],  # Inferior derecha
                puntos_ordenados[3],  # Superior derecha
                puntos_ordenados[2]   # Superior izquierda
            ], dtype=np.float32)

        return puntos_ordenados
    

    # ------------------------------------------------------------
    # 3. Completado de puntos para la Homografia y patron interno
    # ------------------------------------------------------------

    def completar_cuadrado_con_3_puntos(self, puntos):
        if puntos.shape[2] != 3:
            # Se necesitan al menos 3 puntos para completar el cuadrado.
            raise ValueError("")

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
            return circles[0]  # Devuelve el primer círculo detectado 
        else:
            # No se detectaron patrones internos
            return None

    # -------------------------------
    # 4. Ejecucion principal
    # -------------------------------

    def execute(self, verbose: bool = False):
        if not self.images:
            return

        cont = 0
        resultados = []
        for nombre, imagen in zip(self.images_names, self.images):
            try:
                # 1. Preprocesamiento de la imagen
                imagen = cv2.bilateralFilter(imagen, d=9, sigmaColor=75, sigmaSpace=75)

                # 2. Umbralizar la imagen para detectar el color rojo
                img_umbralizada = self._umbralizado(imagen, ajuste_factor=0.3, filtro_tamano=3)

                # 3. Encontrar el marco más grande
                marco_grande = self._marco_grande(img_umbralizada)

                if marco_grande.shape[0] == 3:
                    marco_grande = self.completar_cuadrado_con_3_puntos(marco_grande)

                if marco_grande.shape[0] == 0:
                    # ERROR] No se detectaron puntos.
                    continue

                elif marco_grande.shape[0] == 4:
                    # 4. Detectar patrón interno (si existe)
                    patron_interno = self.detectar_patron_interno(imagen, marco_grande)

                    if patron_interno is not None:
                        marco_grande = self._ordenar_esquinas_con_patron(marco_grande, patron_interno)
                    else:
                        marco_grande = self._ordenar_esquinas(marco_grande)

                    # 6. Calcular la homografía
                    width = self.template_img.shape[1]
                    height = self.template_img.shape[0]
                    puntosTemplate = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

                    H_template2image = cv2.getPerspectiveTransform(puntosTemplate, marco_grande)

                else:
                    H_template2image =self._homografia_fija()

                # 7. Calcular la matriz de proyección y dibujar el cubo
                P = self._calculate_P(H_template2image)
                imagen_final = self._plot_axis_cube_image(imagen, P, False)

                # 8. Guardar solo la imagen final
                resultados.append((nombre, imagen_final, P.copy()))

            except Exception as e:
                # [ERROR] Imagen 
                continue

        return resultados


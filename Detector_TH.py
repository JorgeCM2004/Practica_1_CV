import cv2
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from Algorithm import Algorithm
from itertools import product
from itertools import combinations
from tqdm import tqdm
import math


class Detector_TH(Algorithm):
    def __init__(self, test_path = None, models_path = None):
        super().__init__(test_path, models_path)

    def _intersection(self, line1, line2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        # Ecuaciones de las rectas: y = mx + b
        m1 = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
        b1 = y1 - m1 * x1 if m1 != float('inf') else x1  # b = y - mx

        # m2, b2 para la segunda línea
        m2 = (y4 - y3) / (x4 - x3) if x4 != x3 else float('inf')
        b2 = y3 - m2 * x3 if m2 != float('inf') else x3

        # Si las líneas son paralelas, no tienen intersección
        if m1 == m2:
            return None

        if m1 != float('inf') and m2 != float('inf'):
            # Resolver para x y y
            x_intersect = (b2 - b1) / (m1 - m2)
            y_intersect = m1 * x_intersect + b1
        elif m1 == float('inf'):
            # Línea 1 es vertical
            x_intersect = x1
            y_intersect = m2 * x_intersect + b2
        else:
            # Línea 2 es vertical
            x_intersect = x3
            y_intersect = m1 * x_intersect + b1

        return (int(x_intersect), int(y_intersect))

    def _find_cuadrilatero(self, cuadrilateros, img, t_img):
        best_c = None
        best_score = float('inf')
        resultado_final = None

        width = t_img.shape[1]  # Ancho de la imagen template
        height = t_img.shape[0]  # Alto de la imagen template
        puntosTemplate = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

        for cuadrilatero in tqdm(cuadrilateros, desc= "Procesando cuadriláteros"):
            puntosImagen = np.float32(cuadrilatero)
            M = cv2.getPerspectiveTransform(puntosImagen, puntosTemplate)
            resultado = cv2.warpPerspective(img, M, (width, height))

            # Convertimos a escala de grises
            gray_img = cv2.cvtColor(resultado, cv2.COLOR_BGR2GRAY)
            gray_template = cv2.cvtColor(t_img, cv2.COLOR_BGR2GRAY)

            # diff = cv2.absdiff(gray_template, gray_img)
            # score = np.sum(diff)
            score = np.mean((gray_template - gray_img) ** 2) # Utiliza MSE para calcular la diff 

            if score < best_score:
                best_score = score
                best_c = cuadrilatero
                resultado_final = resultado
            break
            
        if resultado_final is not None:
            img_with_selection = img.copy()
            best_c = np.array(best_c, dtype=np.int32)  # Convertir a entero
            cv2.polylines(img_with_selection, [best_c], isClosed=True, color=(0, 0, 255), thickness=5)

            # Mostrar la imagen con el cuadrilátero seleccionado
            plt.imshow(cv2.cvtColor(img_with_selection, cv2.COLOR_BGR2RGB))
            plt.title("Cuadrilátero Seleccionado")
            plt.show()
        else:
            print("No hay cuadrilatero válido")

    def _agrupaPuntos(self, intersecciones):
        centro_x = sum(pt[0] for pt in intersecciones) / len(intersecciones)
        centro_y = sum(pt[1] for pt in intersecciones) / len(intersecciones)

        centro = [centro_x, centro_y]

        grupos = [[],[],[],[]]
        # Grupo 0 -> Esquina inferior izquierda
        # Grupo 1 -> Esquina inferior derecha
        # Grupo 2 -> Esquina superior izquierda
        # Grupo 3 -> Esquina superior derecha

        for pt in intersecciones:
            if pt[0] < centro_x and pt[1] < centro_y:
                grupos[0].append(pt)
            elif pt[0] > centro_x and pt[1] < centro_y:
                grupos[1].append(pt)
            elif pt[0] < centro_x and pt[1] > centro_y:
                grupos[2].append(pt)
            elif pt[0] > centro_x and pt[1] > centro_y:
                grupos[3].append(pt)
        
        return grupos
    
    def _agrupaPuntosAngulos(self, intersecciones):
        centro_x = sum(pt[0] for pt in intersecciones) / len(intersecciones)
        centro_y = sum(pt[1] for pt in intersecciones) / len(intersecciones)

        grupos = {
            "superior_izquierda": [],
            "superior_derecha": [],
            "inferior_izquierda": [],
            "inferior_derecha": []
        }

        for x, y in intersecciones:
            dx = x - centro_x
            dy = y - centro_y
            angulo = math.atan2(dy, dx)  # Ángulo en radianes

            if dx < 0 and dy < 0:  # Superior izquierda
                grupos["superior_izquierda"].append(((x, y), angulo))
            elif dx > 0 and dy < 0:  # Superior derecha
                grupos["superior_derecha"].append(((x, y), angulo))
            elif dx < 0 and dy > 0:  # Inferior izquierda
                grupos["inferior_izquierda"].append(((x, y), angulo))
            elif dx > 0 and dy > 0:  # Inferior derecha
                grupos["inferior_derecha"].append(((x, y), angulo))
            
        for key in grupos:
            grupos[key] = sorted(grupos[key], key=lambda p: p[1])  # Ordenar por ángulo

        # Devolver solo los puntos (sin los ángulos)
        return {key: [p[0] for p in grupos[key]] for key in grupos}

    def _crearGruposFactibles2(self, grupos):
        sup_izq = grupos.get("superior_izquierda", [])
        sup_der = grupos.get("superior_derecha", [])
        inf_izq = grupos.get("inferior_izquierda", [])
        inf_der = grupos.get("inferior_derecha", [])

        # Generar combinaciones válidas (un punto de cada grupo)
        combinaciones = list(product(sup_izq, sup_der, inf_izq, inf_der))
        return combinaciones

    def _crearGruposFactibles(self, grupos):
        return list(product(grupos[2], grupos[3], grupos[0], grupos[0]))

    def _filtradoAtipicos(self, pts):
        coords_x = [pt[0] for pt in pts]
        coords_y = [pt[1] for pt in pts]

        q1_x, q3_x = np.percentile(coords_x, [25, 75]) # Primer y Tercer Cuartil para RI
        q1_y, q3_y = np.percentile(coords_y, [25, 75]) 

        RI_x = q3_x - q1_x
        RI_y = q3_y - q1_y

        lim_inf_x = q1_x - 1.5 * RI_x
        lim_inf_y = q1_y - 1.5 * RI_y

        lim_sup_x = q3_x + 1.5 * RI_x
        lim_sup_y = q3_y + 1.5 * RI_y

        puntos_filtrados = []
        for x, y in pts:
            if lim_inf_x <= x <= lim_sup_x and lim_inf_y <= y <= lim_sup_y:
                puntos_filtrados.append((x, y))
        return puntos_filtrados

    def _marcorojo(self, color="blanco", imagen_original):
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
            }
        }

        if color not in colores: # Excepcion por pasar mal el color
            raise ValueError(f"Color '{color}' no soportado. Usa 'rojo' o 'blanco'.")


        # print(alto, ancho, canales)
        hsv = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2HSV) # Pasamos a brillo , saturacion, ... para hacer mas facil el umbralizado

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
        
        resultado = cv2.bitwise_and(imagen_original, imagen_original, mask=mask)

        gris = cv2.cvtColor(resultado, cv2.COLOR_BGR2GRAY) # PASO LA IMAGEN A GRISES

        imgCanny = cv2.Canny(gris, 50, 150)

        return imgCanny   

    def _puntos_similares(self, puntos1, puntos2, umbral=10):
        comun = []

        for x1, y1 in puntos1:
            for x2, y2 in puntos2:
                dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if dist <= umbral:
                    comun.append((x1,y1))
                    break
        return comun
    
    def _verPuntos(self, intersecciones, imagen, color):
        for puntos in intersecciones:
            cv2.circle(imagen, puntos, 5, color, -1)  # Círculo rojo Intersección

    def _ordenar_puntos_con_homografia(self, puntos, template_img):
        """
        Ordena los puntos detectados utilizando la homografía calculada con el template.
        :param puntos: Lista de puntos detectados.
        :param template_img: Imagen del template para calcular la homografía.
        :return: Puntos ordenados.
        """
        if len(puntos) < 4:
            raise ValueError("Se necesitan al menos 4 puntos para calcular la homografía.")

        # Definir los puntos del template
        h, w = template_img.shape[:2]
        template_corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])

        # Calcular la homografía
        puntos = np.float32(puntos).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(puntos, template_corners, cv2.RANSAC, 5.0)

        if H is None:
            raise ValueError("No se pudo calcular la homografía para ordenar los puntos.")

        # Transformar los puntos detectados al espacio del template
        puntos_ordenados = cv2.perspectiveTransform(puntos, H)
        return puntos_ordenados.reshape(-1, 2)

    def execute(self):
        if not self.images:
            print("¡¡ No hay imágenes para procesar.")
            return

        resultados = []
        for nombre, imagen in zip(self.images_names, self.images):
            try:
                # 1. Mejora de la imagen
                imagen = cv2.bilateralFilter(imagen, d=9, sigmaColor=75, sigmaSpace=75)

                # 2. Umbralizar la imagen para detectar el color rojo
                img_umbralizada = self._umbralizado(imagen, "rojo")

                # 3. Encontrar el marco más grande
                marco_grande = self._marco_grande(img_umbralizada)

                if marco_grande.shape[0] == 3:
                    marco_grande = self.completar_cuadrado_con_3_puntos(marco_grande)

                if marco_grande.shape[0] == 0:
                    print("NINGÚN punto detectado.")
                    continue

                elif marco_grande.shape[0] != 4:
                    print(f"¡¡ No se han podido obtener los 4 puntos del marco.")
                    continue

                # 4. Detectar patrón interno (si existe)
                patron_interno = self.detectar_patron_interno(imagen, marco_grande)

                if patron_interno is not None:
                    marco_grande = self._ordenar_esquinas_con_patron(marco_grande, patron_interno)
                else:
                    marco_grande = self._ordenar_esquinas(marco_grande)

                # 5. Dibujar el contorno en la imagen original
                COLOR_AZUL = (255, 0, 0)
                cv2.drawContours(imagen, [marco_grande.astype(int)], -1, COLOR_AZUL, 2)

                # 6. Evaluar la homografía
                width = self.template_img.shape[1]  # Ancho de la imagen template
                height = self.template_img.shape[0]  # Alto de la imagen template
                puntosTemplate = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

                puntosImagen = marco_grande.astype(np.float32)
                H_template2image = cv2.getPerspectiveTransform(puntosTemplate, puntosImagen)

                # 7. Calcular la matriz de proyección y dibujar el cubo
                P = self._calculate_P(H_template2image)
                self._plot_axis_cube_image(imagen, P, True)

                resultados.append((imagen, self._plot_axis_cube_image(imagen, P, True), P.copy()))

            except Exception as e:
                print("¡¡ Ocurrió un error durante el procesamiento: ", e)

        return resultados



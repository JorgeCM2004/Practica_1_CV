import cv2
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from Algorithm import Algorithm
from itertools import product
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

    def _marcorojo(self):
        imagen_original = self.images[1]
        nombre_imagen = self.images_names[1]

        alto, ancho, canales = imagen_original.shape

        # print(alto, ancho, canales)
        hsv = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2HSV) # Pasamos a brillo , saturacion, ... para hacer mas facil el umbralizado

        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 50, 255])

        mask = cv2.inRange(hsv, lower_white, upper_white)
        
        resultado = cv2.bitwise_and(imagen_original, imagen_original, mask=mask)

        resultado_rgb = cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB)

        gris = cv2.cvtColor(resultado_rgb, cv2.COLOR_BGR2GRAY) # PASO LA IMAGEN A GRISES

        imgCanny = cv2.Canny(gris, 50, 150)

        return imgCanny   



    def execute(self):
        imagen = self.images[1].copy()    
        # gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises

        # edges = cv2.Canny(gray, 50, 150)
        
        # Apaño con lo de alberto
        edges = self._marcorojo().copy() # Edges el nombre de la imagen

        # 1- OBTENER LAS RECTAS
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=250, minLineLength=200, maxLineGap=30)
        # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=300, minLineLength=200, maxLineGap=30)

        # 2- OBTENER LOS PUNTOS DE INTERSECCION 

        alto, ancho, _ = imagen.shape
        intersecciones = []
        for i in range(len(lines)):
            x1, y1, x2, y2 = lines[i][0]
            cv2.line(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Líneas verdes

            for e in range(i + 1, len(lines)): 
                line1 = lines[i][0]
                line2 = lines[e][0]  
                intersection_point = self._intersection(line1, line2)

                if intersection_point and 0 <= intersection_point[0] < ancho and 0 <= intersection_point[1] < alto:
                    intersecciones.append(intersection_point)

        intersecciones = self._filtradoAtipicos(intersecciones)

        for puntos in intersecciones:
           cv2.circle(imagen, puntos, 5, (0, 0, 255), -1)  # Círculo rojo Intersección

        # 3- ORDENAR LOS PUNTOS por ejes
        print("Puntos de Interseccion: ", len(intersecciones))
        
        # grupos = self._agrupaPuntos(intersecciones)
        # combinaciones = self._crearGruposFactibles(grupos)
        # print("Combinaciones: ", len(combinaciones))

        # ORGANIZAR CON ANGULOS
        grupos = self._agrupaPuntosAngulos(intersecciones)
        combinaciones = self._crearGruposFactibles2(grupos)
        print("Combinaciones: ", len(combinaciones))

        # 4- HOMOGRAFIA para corregir perspectiva
        self._find_cuadrilatero(combinaciones[0:], imagen, self.template_img)

        
        # Mostrar la imagen con las líneas detectadas
        # plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
        # plt.title("Líneas detectadas con la Transformada de Hough")
        # plt.show()      
        





Detector_TH(None, None).execute()



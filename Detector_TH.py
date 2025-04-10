import cv2
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from Algorithm import Algorithm
from itertools import combinations
from tqdm import tqdm


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

            diff = cv2.absdiff(gray_template, gray_img)
            score = np.sum(diff)

            if score < best_score:
                best_score = score
                best_c = cuadrilatero
                resultado_final = resultado
            
        if resultado_final is not None:
            img_with_selection = img.copy()
            best_c = np.array(best_c, dtype=np.int32)  # Convertir a entero
            cv2.polylines(img_with_selection, [best_c], isClosed=True, color=(0, 255, 0), thickness=3)

            # Mostrar la imagen con el cuadrilátero seleccionado
            plt.imshow(cv2.cvtColor(img_with_selection, cv2.COLOR_BGR2RGB))
            plt.title("Cuadrilátero Seleccionado")
            plt.show()
        else:
            print("No hay cuadrilatero válido")


    def execute(self):
        imagen = self.images[0].copy()    
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises

        edges = cv2.Canny(gray, 50, 150)
        
        # 1- OBTENER LAS RECTAS
        # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=250, minLineLength=300, maxLineGap=40)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=300, minLineLength=300, maxLineGap=40)

        # 2- OBTENER LOS PUNTOS DE INTERSECCION 
        cont = 0
        intersecciones = []
        for i in range(len(lines)):
            x1, y1, x2, y2 = lines[i][0]
            #cv2.line(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Líneas verdes

            for e in range(i + 1, len(lines)): 
                line1 = lines[i][0]
                line2 = lines[e][0]  
                intersection_point = self._intersection(line1, line2)

                if intersection_point:
                    intersecciones.append(intersection_point)
                    cont += 1

        # for puntos in intersecciones:
        #    cv2.circle(imagen, puntos, 5, (0, 0, 255), -1)  # Círculo rojo Intersección

        # 3- ORDENAR LOS PUNTOS por ejes
        intersecciones = sorted(intersecciones, key=lambda p: p[0]) # Ordenamos por X

        pts_arriba = sorted(intersecciones[:2], key=lambda p: p[1])  # Ordenamos por Y
        pts_abajo = sorted(intersecciones[2:], key=lambda p: p[1])  

        # Puntos para los cuadrilateros 
        ordered_pts = pts_arriba + pts_abajo

        cuadrilateros = list(combinations(ordered_pts, 4))
        print(cuadrilateros[:5])

        # 4- HOMOGRAFIA para corregir perspectiva
        # self._find_cuadrilatero(cuadrilateros, imagen, self.template_img)

        # Mostrar la imagen con las líneas detectadas
        # print("Numero de Intersecciones:", cont)
        # plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
        # plt.title("Líneas detectadas con la Transformada de Hough")
        # plt.show()      
        





Detector_TH(None, None).execute()



import cv2
import numpy as np
from Algorithm import Algorithm
import copy


class MarcoRojo(Algorithm):
    def __init__(self, test_path=None, models_path=None):
        super().__init__(test_path, models_path)

    def execute(self):
        resultados = []
        for nombre, imagen in zip(self.images_names, self.images):
            try:
                
                imagen_umbralizada = self.umbralizado_rojo(imagen)

                marco_grande = self.marco_grande(imagen_umbralizada)

                if marco_grande.shape[0] == 3:
                    marco_grande = self.completar_cuadrado_con_3_puntos(marco_grande)

                if marco_grande.shape[0] != 4:
                    print(f"[ERROR] Imagen '{nombre}': no se han podido obtener los 4 puntos del marco.")
                    continue
                
                destiny_points = []
                if marco_grande.size != 0:
                    marco_grande = self.ordenar_esquinas(marco_grande)
                    etiquetas_marco_1 = ["0", "1", "2", "3"]
                    for i, punto in enumerate(marco_grande):
                        x, y = int(punto[0]), int(punto[1])
                        destiny_points.append([x,y]) # np array s.type = f32
                        cv2.circle(imagen, (x, y), 5, (0, 0, 255), -1)
                        cv2.putText(imagen, etiquetas_marco_1[i], (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                

                template_points = np.array([[0, 0], [self.template_img.shape[1], 0], [0, self.template_img.shape[0]], [self.template_img.shape[1],  self.template_img.shape[0]]], dtype = np.float32) # Píxeles (horizontal, vertical)
                destiny_points = np.array([destiny_points[0], destiny_points[1], destiny_points[2],  destiny_points[3]], dtype = np.float32)
                
                H_template2image = cv2.getPerspectiveTransform(template_points, destiny_points)

                P = self._calculate_P(H_template2image)

                resultados.append((imagen, self._plot_axis_cube_image(imagen, P, True), P.copy()))


            except Exception as e:
                print(f"[ERROR] Imagen '{nombre}': {e}")
                continue

        return resultados
    

    def marco_grande(self, imagen):
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) # PASO LA IMAGEN A GRISES

        imgCanny = cv2.Canny(gris, 50, 150)

        contornos, jerarquia = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Encontramos los contornos
        '''
        Modos de contorno que se pueden usar retr_(External, list, ccomp, tree)
        Para recuperacion de contornos hay dos:
            - CHAIN_APPROX_SIMPLE no almacena los redundantes
            - RETR_EXTERNAL solo recupera los contornos exteriores
        '''
        alto, ancho, _ = imagen.shape

        marco_grande, _ = self.biggestContour(contornos, alto, ancho)

        return marco_grande


    def umbralizado_rojo(self, imagen):
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV) # Pasamos a brillo , saturacion, ... para hacer mas facil el umbralizado

        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        mask = cv2.bitwise_or(mask1, mask2)

        resultado = cv2.bitwise_and(imagen, imagen, mask=mask)

        resultado_rgb = cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB)

        return resultado_rgb


    def biggestContour(self, contours, alto, ancho):
        biggest = np.array([])
        max_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area > (alto * ancho * 0.05):
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area
        return biggest, max_area
    


    def ordenar_esquinas(self, puntos):
        puntos = puntos.reshape((4, 2))
        puntos_ordenados = np.zeros((4, 2), dtype=np.float32)

        suma = puntos.sum(axis=1)
        resta = np.diff(puntos, axis=1)
    
        puntos_ordenados[0] = puntos[np.argmin(suma)]  # superior izquierda
        puntos_ordenados[3] = puntos[np.argmax(suma)]  # inferior derecha
        puntos_ordenados[1] = puntos[np.argmin(resta)] # superior derecha
        puntos_ordenados[2] = puntos[np.argmax(resta)] # inferior izquierda

        return puntos_ordenados
    

    def completar_cuadrado_con_3_puntos(self, puntos):
        if puntos.shape[0] != 3:
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


MarcoRojo(None, None).execute()

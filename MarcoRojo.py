import cv2
import numpy as np
from Algorithm import Algorithm


class MarcoRojo(Algorithm):
    def __init__(self, test_path=None, models_path=None):
        super().__init__(test_path, models_path)

    def execute(self):
        imagen_original = self.images[0]
        hsv = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2HSV) # Pasamos a brillo , saturacion, ... para hacer mas facil el umbralizado

        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        mask = cv2.bitwise_or(mask1, mask2)
        
        resultado = cv2.bitwise_and(imagen_original, imagen_original, mask=mask)

        resultado_rgb = cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB)

#----------------------------------------------------------------- UMBRALIZADO DEL ROJO DEL MARCO
        gris = cv2.cvtColor(resultado_rgb, cv2.COLOR_BGR2GRAY) # PASO LA IMAGEN A GRISES

        imgCanny = cv2.Canny(gris, 50, 150)

        contornos, jerarquia = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Encontramos los contornos
        ''' 
        Modos de contorno que se pueden usar retr_(External, list, ccomp, tree)
        Para recuperacion de contornos hay dos:
            - CHAIN_APPROX_SIMPLE no almacena los redundantes
            - RETR_EXTERNAL solo recupera los contornos exteriores
        '''
        marco, _ = self.biggestContour(contornos)
        print(marco)

        print(contornos)

        
        if marco.size != 0:
            marco = self.ordenar_esquinas(marco)
            etiquetas = ["0", "1", "2", "3"]
            for i, punto in enumerate(marco):
                x, y = int(punto[0][0]), int(punto[0][1])
                cv2.circle(imagen_original, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(imagen_original, etiquetas[i], (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
    
        
        escala = 0.25
        altura, ancho = imagen_original.shape[:2]
        imagen_redimensionada = cv2.resize(imagen_original, (int(ancho * escala), int(altura * escala)))
        
        cv2.imshow("Puntos en las esquinas", imagen_redimensionada)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def biggestContour(self, contours):
        biggest = np.array([])
        max_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area > 5000:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area
        return biggest, max_area
    
    def ordenar_esquinas(self, esquinas):
        esquinas = esquinas.reshape((4, 2))
        suma = esquinas.sum(axis=1)
        diferencia = np.diff(esquinas, axis=1)
        ordenadas = np.zeros((4, 2), dtype=np.int32)
        ordenadas[0] = esquinas[np.argmin(suma)]  
        ordenadas[2] = esquinas[np.argmax(suma)] 
        ordenadas[1] = esquinas[np.argmin(diferencia)]
        ordenadas[3] = esquinas[np.argmax(diferencia)]  
        return ordenadas.reshape((4, 1, 2))


MarcoRojo(None, None).execute()
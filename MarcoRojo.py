import cv2
import numpy as np
#import matplotlib.pyplot as plt
from copy import copy
from Algorithm import Algorithm

class MarcoRojo(Algorithm):
    def __init__(self, test_path = None, models_path = None):
        super().__init__(test_path, models_path)


    def execute(self):

        imagen_original = self.template_img.copy()        
        gris = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY)
        _, imagen_umbralizada = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY)

        imgCanny = cv2.Canny(imagen_umbralizada, 50, 150)
        

        #Encontra contornos
        contornos,jerarquia = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Modos de contorno que se pueden usar retr_(External, list, ccomp, tree)
        #de recuperacion de contornos hay dos, CHAIN_APPROX_SIMPLE no almacena los redundantgesy
        #RETR_EXTERNAL solo recupera los contornos exteriores
        #print(contornos) # los puntos de los contornos, tantos como contornos hay
        print(jerarquia)
        contornos_filtrados = [c for c in contornos if cv2.arcLength(c, True) > 120]  # Ajusta el umbral

        contornos_filtrados_2 = [c for c in contornos if cv2.contourArea(c) > 10]  # Ajusta el umbral según el ruido



        cv2.drawContours(imagen, contornos_filtrados, -1, (0,255,0), 3) #el -1 es paera mostrar todos los contornos, el 3 es el grosor de la linea


        cv2.imshow("Hola", imagen)
        #cv2.imshow("Hola", imgCanny)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def biggestContour(self, contours):
        biggest = np.array([])
        max_area = 0
        for i in contours:
            area = cv2.contourArea(i)
            if area > 5000:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.02 * peri, True)
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area
        return biggest,max_area






MarcoRojo(None, None).execute()











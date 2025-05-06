import cv2
import numpy as np
import glob

def calibrar_camara(ruta_imagenes, tamano_tablero=(9, 6), tamano_casilla=1.0):
    """
    Calibra la cámara usando imágenes de un tablero de ajedrez.

    Args:
        ruta_imagenes (str): Ruta a las imágenes del tablero de ajedrez.
        tamano_tablero (tuple): Número de esquinas internas del tablero (filas, columnas).
        tamano_casilla (float): Tamaño de cada casilla del tablero en unidades reales (por ejemplo, milímetros).

    Returns:
        tuple: Matriz de intrínsecos (K), coeficientes de distorsión, y otros parámetros.
    """
    # Preparar puntos del mundo real (0,0,0), (1,0,0), (2,0,0), ..., según el tamaño del tablero
    objp = np.zeros((tamano_tablero[0] * tamano_tablero[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:tamano_tablero[1], 0:tamano_tablero[0]].T.reshape(-1, 2)
    objp *= tamano_casilla

    # Listas para almacenar puntos del mundo real y puntos de la imagen
    objpoints = []  # Puntos 3D en el mundo real
    imgpoints = []  # Puntos 2D en la imagen

    # Leer todas las imágenes del tablero de ajedrez
    imagenes = glob.glob(ruta_imagenes)

    for fname in imagenes:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Encontrar las esquinas del tablero de ajedrez
        ret, corners = cv2.findChessboardCorners(gray, tamano_tablero, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Dibujar y mostrar las esquinas
            cv2.drawChessboardCorners(img, tamano_tablero, corners, ret)
            cv2.imshow('Esquinas detectadas', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Calibrar la cámara
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return K, dist, rvecs, tvecs

def aproximar_intrinsecos(ancho_imagen, alto_imagen, distancia_focal_px=None):
    """
    Aproxima la matriz de intrínsecos de la cámara.

    Args:
        ancho_imagen (int): Ancho de la imagen en píxeles.
        alto_imagen (int): Alto de la imagen en píxeles.
        distancia_focal_px (float): Distancia focal en píxeles (opcional).

    Returns:
        numpy.ndarray: Matriz de intrínsecos (3x3).
    """
    if distancia_focal_px is None:
        distancia_focal_px = max(ancho_imagen, alto_imagen)  # Aproximar como el mayor tamaño de la imagen

    c_x = ancho_imagen / 2
    c_y = alto_imagen / 2

    K = np.array([
        [distancia_focal_px, 0, c_x],
        [0, distancia_focal_px, c_y],
        [0, 0, 1]
    ])

    return K


ruta_imagenes = "/ruta/a/imagenes/*.jpg"  # Ruta a las imágenes del tablero de ajedrez
tamano_tablero = (9, 6)  # Número de esquinas internas del tablero
tamano_casilla = 25.0  # Tamaño de cada casilla en milímetros

K, dist, rvecs, tvecs = calibrar_camara(ruta_imagenes, tamano_tablero, tamano_casilla)
print("Matriz de intrínsecos (K):")
print(K)



ancho_imagen = 1920
alto_imagen = 1080
distancia_focal_px = 1500  # Aproximación

K = aproximar_intrinsecos(ancho_imagen, alto_imagen, distancia_focal_px)
print("Matriz de intrínsecos aproximada:")
print(K)
import cv2
import numpy as np
from transformaciones import TransformacionesEuclideanas, TransformacionesAfines

def detectar_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Poición del cursor (x, y): ({x}, {y})")
        param["clicks"] = param["clicks"] + 1
        param["puntos"].append([x, y])

def obtener_puntos(imagen_vis):
    params = {
        "imagen": imagen_vis, 
        "clicks": 0,
        "puntos": []
    }
    cv2.namedWindow("Imagen", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Imagen", detectar_click, param=params)
    while True:
        cv2.imshow("Imagen", imagen_vis)
        if cv2.waitKey(1) & 0xFF == ord('q') or params["clicks"] >= 3:
            break 
    cv2.destroyWindow("Imagen")
    cv2.waitKey(1)
    return params["puntos"]


ruta = "Modulo_02/img/lena.jpeg"
#imagen = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
imagen = cv2.imread(ruta)
h,w = imagen.shape[0:2]

centro_y, centro_x = int(np.floor(h/2)), int(np.floor(w/2))
T1 = TransformacionesEuclideanas.traslado(-centro_x, -centro_y)
T2 = TransformacionesEuclideanas.traslado(centro_x, centro_y)

RN1 = TransformacionesEuclideanas.rotacion(-np.pi/2,1)
TN1 = TransformacionesEuclideanas.traslado(2, 2)
CN1 = TransformacionesAfines.cizallamiento_horizontal(1.025)

T = T2 @ TN1 @ CN1 @ RN1 @ T1

imagen_procesada = TransformacionesAfines.transformacion_opencv(imagen, T[:2,:], (w,h))

puntos_origen = obtener_puntos(imagen).copy()
puntos_destino = obtener_puntos(imagen_procesada).copy()
M = cv2.getAffineTransform(np.float32(np.array(puntos_origen)), np.float32(np.array(puntos_destino)))
imagen_estimada = cv2.warpAffine(imagen, M, (w,h))

cv2.imshow("ImagenEstimada", imagen_estimada)
cv2.waitKey(0)

cv2.destroyAllWindows()
cv2.waitKey(1)



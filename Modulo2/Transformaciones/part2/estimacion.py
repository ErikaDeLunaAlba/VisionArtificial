import cv2
import numpy as np

def detectar_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Posición (x, y): ({x}, {y})")
        param["puntos"].append([x, y])
        param["clicks"] += 1

def obtener_puntos(imagen, nombre_ventana):
    params = {
        "puntos": [],
        "clicks": 0
    }

    cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(nombre_ventana, detectar_click, param=params)

    print(f"Selecciona 3 puntos de referencia en {nombre_ventana}")

    while True:
        cv2.imshow(nombre_ventana, imagen)
        if cv2.waitKey(1) & 0xFF == ord('q') or params["clicks"] >= 3:
            break

    cv2.destroyWindow(nombre_ventana)
    cv2.waitKey(1)

    return np.array(params["puntos"], dtype=np.float32)


imagen_original = cv2.imread("cameraman.png")
imagen_final = cv2.imread("cameraman_processed.png")

h, w = imagen_original.shape[:2]


puntos_origen = obtener_puntos(imagen_original, "Imagen Original")
puntos_destino = obtener_puntos(imagen_final, "Imagen Final")


M = cv2.getAffineTransform(puntos_origen, puntos_destino)

print("\nMatriz afín Estimada")
print(M)


imagen_estimada = cv2.warpAffine(imagen_original, M, (w, h))


cv2.imshow("Imagen Original", imagen_original)
cv2.destroyWindow("Imagen Original")
cv2.imshow("Imagen Final", imagen_final)
cv2.destroyWindow("Imagen Final")
cv2.imshow("Imagen Estimada", imagen_estimada)

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
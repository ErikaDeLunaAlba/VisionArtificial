import cv2
import numpy as np

def ecualizar_camara(imagen):
    imagen_yuv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    canal_y = imagen_yuv[:,:,0]
    canal_y_ecualizado = cv2.equalizeHist(canal_y)
    imagen_yuv[:,:,0] = canal_y_ecualizado
    return cv2.cvtColor(imagen_yuv, cv2.COLOR_HSV2BGR)

# Captura desde la fuente default
camara = cv2.VideoCapture(0)

print("Oprime q para salir de la captura de camara")
while camara.isOpened():
    while True:
        ret, frame = camara.read()
        if not ret:
            print("Error al leer frame desde la cámara")
            break
        frame_eq = ecualizar_camara(frame)
        frame_comparativo = np.hstack((frame, frame_eq))
        cv2.imshow("Camara", frame_comparativo)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    break

camara.release()
cv2.destroyAllWindows()
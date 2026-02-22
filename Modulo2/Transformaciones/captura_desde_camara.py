import cv2
import numpy as np
from transformaciones import TransformacionesEuclideanas

# Captura desde la fuente default
camara = cv2.VideoCapture(0)
w = int(camara.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(camara.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Imagen {h} x {w}")

centro_y, centro_x = int(round(h/2)), int(round(w/2))

R = TransformacionesEuclideanas.rotacion(-np.pi, 1.5)
T1 = TransformacionesEuclideanas.traslado(-centro_x, -centro_y)
T2 = TransformacionesEuclideanas.traslado(centro_x, centro_y)

#T = traslado2 @ rotacion @ traslado @ es producto matricial
T = T2 @ R @ T1
M_inv = np.linalg.inv(T)

while camara.isOpened():
    while True:
        ret, frame = camara.read()
        if not ret:
            print("Error al leer frame desde la cámara")
            break

        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #frame = TransformacionesEuclideanas.transformacion(frame, M_inv, (w,h))
        frame = TransformacionesEuclideanas.transformacion_opencv(frame, T[0:2,:],(w,h))

        print(frame.shape)
        cv2.imshow("Camara", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    break

camara.release()
cv2.destroyAllWindows()

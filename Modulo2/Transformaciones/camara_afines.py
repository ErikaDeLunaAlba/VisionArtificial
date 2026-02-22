import cv2
import numpy as np
from transformaciones import TransformacionesEuclideanas
from transformaciones import TransformacionesAfines

# Captura desde la fuente default
camara = cv2.VideoCapture(0)
w = int(camara.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(camara.get(cv2.CAP_PROP_FRAME_HEIGHT))
centro_y, centro_x = int(round(h/2)), int(round(w/2))

C1 = TransformacionesAfines.dilatacion_no_uniforme(0.5, 1)
T1 = TransformacionesEuclideanas.traslado(-centro_x, -centro_y)
T2 = TransformacionesEuclideanas.traslado(centro_x, centro_y)

T = T2 @ C1 @ T1
while camara.isOpened():
    while True:
        ret, frame = camara.read()
        if not ret:
            print("Error al leer frame desde la cámara")
            break
        frame = TransformacionesAfines.transformacion_opencv(frame, T[0:2,:], (w,h))
        cv2.imshow("Camara", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    break
camara.release()
cv2.destroyAllWindows()

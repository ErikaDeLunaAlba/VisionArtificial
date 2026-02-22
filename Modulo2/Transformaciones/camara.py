import cv2
# Captura desde la fuente default
camara = cv2.VideoCapture(0)

print("Oprime q para salir de la captura de camara")
while camara.isOpened():
    while True:
        ret, frame = camara.read()
        if not ret:
            print("Error al leer frame desde la cámara")
            break
        cv2.imshow("Camara", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    break

camara.release()
cv2.destroyAllWindows()
import cv2
import numpy as np

def spectrum(img):
    img = np.float32(img)
    img_dft =  cv2.dft(img, flags = cv2.DFT_COMPLEX_OUTPUT)
    img_dft_shifted = np.fft.fftshift(img_dft)
    img_mag = cv2.magnitude(img_dft_shifted[:,:,0], img_dft_shifted[:,:,1])
    return np.log(img_mag + 1)

# Captura desde la fuente default
camara = cv2.VideoCapture(0)

print("Oprime q para salir de la captura de camara")
while camara.isOpened():
    while True:
        ret, frame = camara.read()
        if not ret:
            print("Error al leer frame desde la cámara")
            break

        # Obtener espectro y mostrarla
        frame_y = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_fft = spectrum(frame_y)
        
        # Se normaliza y cambia a tipo uint8. cv2.imshow necesita este formato
        # de la imagen para mostarlo (de 0 a 255)
        cv2.normalize(frame_fft, frame_fft, 0, 255, cv2.NORM_MINMAX)
        mag = np.uint8(frame_fft)
        
        frame_show = np.hstack((frame_y, mag))


        cv2.imshow("Camara", frame_show)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    break

camara.release()
cv2.destroyAllWindows()

"""
Tarea: 
Probar con cuatro imágenes de letras.
Barras horizontales
Barras Verticales
Una malla

Preguntas:
¿Qué se observa con cada imagen en el espectro?
¿Qué pasa con el espectro si vas girando la imagen?
¿Cómo crees que la transformada de Fourier pueda ayudar a detectar formas y cambios de dirección en imágenes?
"""
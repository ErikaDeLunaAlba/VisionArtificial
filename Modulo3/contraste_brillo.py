import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class Ventana(tk.Tk):
    def __init__(self):
        super().__init__(className="Control de contraste y brillo")

        self.title = "Control de contraste y brillo"
        self.geometry("1100x500")
        self.protocol("WM_DELETE_WINDOW", self.cierre)

        self.ruta_imagen = "img/cameraman.png"

        self.imagen = cv2.imread(self.ruta_imagen)
        self.h, self.w = self.imagen.shape[0:2]
        self.imagen_pl = Image.open(self.ruta_imagen)
        self.imagen_tk = ImageTk.PhotoImage(self.imagen_pl)
        self.imagen_procesada = cv2.imread(self.ruta_imagen)
        self.imagen_procesada_pl = Image.open(self.ruta_imagen)
        self.imagen_procesada_tk = ImageTk.PhotoImage(self.imagen_procesada_pl)

        self.fig, self.ax = plt.subplots(figsize=(1, 2))
        self.__check_status = tk.BooleanVar()
        self.mostrar_ventana()

    def cierre(self):
        plt.close("all")
        self.quit()
        self.destroy()

    def mostrar_ventana(self):
        self.__canvas_imagen = tk.Canvas(self, width=self.w, height=self.h, background="black")
        self.__canvas_imagen_proc = tk.Canvas(self, width=self.w, height=self.h, background="black")

        if self.imagen is not None:
            self.__canvas_imagen.create_image(self.w/2, self.h/2, image=self.imagen_tk)
            self.__canvas_imagen.place(x=30, y=40)

        self.mostrar_imagen_procesada(0, 1, 1)

        self.__escala_brillo = tk.Scale(self, label="Brillo", from_=-100, to=100, orient=tk.HORIZONTAL, command=self.actualizar_imagen)
        self.__escala_contraste = tk.Scale(self, label="Contraste", from_=-1, variable=tk.IntVar(value=1), to=3, orient=tk.HORIZONTAL, resolution=0.05, command=self.actualizar_imagen)
        self.__escala_gamma = tk.Scale(self, label="Gamma", from_=-2, to=4, variable=tk.IntVar(value=1), orient=tk.HORIZONTAL, resolution=0.05, command=self.actualizar_imagen)
        
        self.__escala_brillo.place(x=100, y=250+50)
        self.__escala_contraste.place(x=250 , y=250+50)
        self.__escala_gamma.place(x=100, y=250+120)

        self.__canvas_hist = FigureCanvasTkAgg(self.fig, master=self)
        self.__canvas_hist.get_tk_widget().place(x=550, y=40, width=500, height=400)
        self.mostrar_histograma(self.obtener_y(self.imagen_procesada))

        self.__canvas_check = tk.Checkbutton(self, text="Ecualizar Histograma", variable=self.__check_status, command=self.actualizar_imagen, onvalue=True)
        self.__canvas_check.place(x=100, y=250+200)

    def mostrar_imagen_procesada(self, brillo, contraste, gamma):
        self.procesar_imagen(contraste, brillo, gamma)
        im = cv2.cvtColor(self.imagen_procesada, cv2.COLOR_BGR2RGB)
        self.imagen_procesada_pl = Image.fromarray(im)
        self.imagen_procesada_tk = ImageTk.PhotoImage(self.imagen_procesada_pl)
        self.__canvas_imagen_proc.create_image(self.w/2, self.h/2, image=self.imagen_procesada_tk)
        self.__canvas_imagen_proc.place(x=250 + 30, y=40)
        
    def actualizar_imagen(self, val=None):
        brillo = self.__escala_brillo.get()
        contraste = self.__escala_contraste.get()
        gamma = self.__escala_gamma.get()
        self.mostrar_imagen_procesada(brillo, contraste, gamma)
        self.mostrar_histograma(self.obtener_y(self.imagen_procesada))

    def obtener_y(self, imagen):
        """Obtiene el canal Y de YUV"""
        return cv2.cvtColor(imagen, cv2.COLOR_BGR2YUV)[:,:,0]

    def mostrar_histograma(self, datos):
        self.ax.clear()
        self.ax.set_title('Histograma de Imagen Procesada (Canal Y)')
        self.ax.set_xlabel('Valor')
        self.ax.set_ylabel('Frecuencia')
        self.ax.hist(datos.flatten(), bins=256, alpha=0.7, range=(0,255), color='blue', edgecolor='black')
        self.__canvas_hist.draw_idle()

    def procesar_imagen(self, contraste, brillo, gamma):
        # Convertir BGR a YUV
        imagen_color_YUV = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2YUV)
        
        # Obtener el canal Y (luminancia)
        imagen_y = imagen_color_YUV[:,:,0]
        
        # Procesar el canal Y
        imagen_y_procesada = Procesador.contraste_brillo_centrado(imagen_y, contraste, brillo)
        imagen_y_procesada = Procesador.correccion_gamma(imagen_y_procesada, gamma)
        
        if self.__check_status.get():
            imagen_y_procesada = Procesador.ecualizar_hist(imagen_y_procesada)
        
        # Actualizar el canal Y procesado
        imagen_color_YUV[:,:,0] = imagen_y_procesada
        
        # Convertir de vuelta a BGR
        self.imagen_procesada = cv2.cvtColor(imagen_color_YUV, cv2.COLOR_YUV2BGR)

class Procesador:
    @staticmethod
    def contraste_brillo(imagen, contraste, brillo):
        imagen_p = imagen.astype(np.float32)    
        mat_proc = np.clip(contraste * imagen_p + brillo, 0, 255).astype(np.uint8)
        return mat_proc
    
    @staticmethod
    def contraste_brillo_centrado(imagen, contraste, brillo):
        imagen_p = imagen.astype(np.float32)    
        mat_proc = np.clip(contraste * (imagen_p - 128) + 128 + brillo, 0, 255).astype(np.uint8)
        return mat_proc
    
    @staticmethod
    def correccion_gamma(imagen, gamma):
        """
        ¿Qué hace gamma visualmente?
        γ < 1
        Aclara sombras
        Expande tonos oscuros
        Comprime altas luces
        Muy útil para imágenes oscuras.
        γ > 1
        Oscurece sombras
        Expande altas luces
        Comprime tonos bajos
        Útil para imágenes muy brillantes.
        """
        return np.clip(((imagen.astype(np.float32)*1/255)**(gamma))*255, 0, 255).astype(np.uint8)
    
    @staticmethod
    def ecualizar_hist(imagen):
        return cv2.equalizeHist(imagen)


if __name__ == "__main__":
    v = Ventana()
    v.mainloop()
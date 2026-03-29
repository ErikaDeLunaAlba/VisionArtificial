import tkinter as tk
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys

def get_edges(thres1, thres2):
    return cv2.Canny(img, float(thres1), float(thres2))

def scroll(val):
    th1 = t1.get()
    th2 = t2.get()
    update_image(get_edges(th1, th2))

def update_image(img_edges):
    ax.clear()  # Limpiar el eje antes de mostrar nueva imagen
    ax.imshow(img_edges, cmap="gray")
    ax.set_title(f"Umbrales: {t1.get():.0f}, {t2.get():.0f}")
    canvas.draw_idle()

def on_closing():
    """Manejar el cierre de la ventana correctamente"""
    plt.close('all')  # Cerrar todas las figuras de matplotlib
    canvas.get_tk_widget().destroy()  # Destruir el widget canvas
    ventana.quit()  # Salir del mainloop
    ventana.destroy()  # Destruir la ventana
    sys.exit(0)  # Salir del programa

# Crear ventana
ventana = tk.Tk()
ventana.geometry("650x700")
ventana.title("Detector de bordes Canny")
ventana.protocol("WM_DELETE_WINDOW", on_closing)  # Configurar manejador de cierre

# Variables para los umbrales
t1 = tk.DoubleVar(ventana, 100)
t2 = tk.DoubleVar(ventana, 100)

# Crear sliders con etiquetas
tk.Label(ventana, text="Umbral 1 (bajo):").pack(pady=(10, 0))
thres1_scroll = tk.Scale(ventana, from_=0, to=500, command=scroll, 
                         orient=tk.HORIZONTAL, variable=t1, length=400)
thres1_scroll.pack(pady=5)

tk.Label(ventana, text="Umbral 2 (alto):").pack(pady=(10, 0))
thres2_scroll = tk.Scale(ventana, from_=0, to=500, command=scroll, 
                         orient=tk.HORIZONTAL, variable=t2, length=400)
thres2_scroll.pack(pady=5)

# Cargar imagen
img = cv2.imread("img/piezas.jpeg", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: No se pudo cargar la imagen")
    ventana.destroy()
    sys.exit(1)

# Crear figura de matplotlib
fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
ax.imshow(img, cmap="gray")
ax.set_title("Imagen original - Mueve los sliders")
ax.axis('off')  # Ocultar ejes para mejor visualización

# Crear canvas
canvas = FigureCanvasTkAgg(fig, ventana)
canvas.draw()
canvas.get_tk_widget().pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

# Mostrar la imagen inicial de bordes (opcional)
update_image(get_edges(100, 100))

ventana.mainloop()
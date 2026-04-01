import cv2
import numpy as np

# =========================
# Parámetros
# =========================
lk_params = dict(
    winSize=(15, 15),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)
)

feature_params = dict(
    maxCorners=200,
    qualityLevel=0.2,
    minDistance=5,
    blockSize=7
)

MAX_ERROR = 20
MAX_DISPLACEMENT = 50  # control de drift
MIN_POINTS_RATIO = 0.2

# =========================
# Funciones
# =========================
def detect_features(gray, mask=None):
    pts = cv2.goodFeaturesToTrack(gray, mask=mask, **feature_params)
    if pts is not None:
        pts = np.float32(pts)
    return pts

def filter_points(p0, p1, st, err):
    """Filtrado robusto de puntos"""
    if p1 is None or st is None or err is None:
        return None, None

    st = st.reshape(-1)
    err = err.reshape(-1)

    valid = (st == 1) & (err < MAX_ERROR)

    p0_valid = p0[valid]
    p1_valid = p1[valid]

    # Control de drift (movimientos absurdos)
    displacement = np.linalg.norm(p1_valid - p0_valid, axis=2).reshape(-1)
    drift_mask = displacement < MAX_DISPLACEMENT

    return p0_valid[drift_mask], p1_valid[drift_mask]

# =========================
# Video
# =========================
video = cv2.VideoCapture("img/Traffic.mp4")

ret, old_frame = video.read()
if not ret:
    raise Exception("No se pudo leer el video")

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# =========================
# Máscara centrada (ROI)
# =========================
h, w = old_gray.shape
corner_mask = np.zeros_like(old_gray, dtype=np.uint8)

# Ajustar valores para centrado y tamaño del ROI
x1 = int(w * 0.4)
x2 = int(w * 0.64)
y1 = int(h * 0.37)
y2 = int(h * 0.65)

corner_mask[y1:y2, x1:x2] = 255

# Mostrar máscara inicial
cv2.imshow("Mascara ROI", corner_mask)

# =========================
# Inicialización de puntos
# =========================
p0 = detect_features(old_gray, corner_mask)

mask_draw = np.zeros_like(old_frame)
colors = np.random.randint(0, 255, (1000, 3))

print("Presiona 'q' para salir | 's' para guardar captura")

# =========================
# Loop principal
# =========================
while True:
    ret, frame = video.read()

    if not ret:
        print("Reiniciando video...")
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # =========================
    # Optical Flow
    # =========================
    if p0 is not None and len(p0) > 0:
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )

        good_old, good_new = filter_points(p0, p1, st, err)

        if good_new is not None and len(good_new) > 0:

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()

                mask_draw = cv2.line(
                    mask_draw,
                    (int(c), int(d)),
                    (int(a), int(b)),
                    colors[i % len(colors)].tolist(),
                    2
                )

                frame = cv2.circle(
                    frame,
                    (int(a), int(b)),
                    4,
                    colors[i % len(colors)].tolist(),
                    -1
                )

            p0 = good_new.reshape(-1, 1, 2)

        else:
            p0 = None

    # =========================
    # Re-detección inteligente
    # =========================
    if p0 is None or len(p0) < feature_params['maxCorners'] * MIN_POINTS_RATIO:
        print("Re-detectando puntos...")

        new_pts = detect_features(frame_gray, corner_mask)

        if new_pts is not None:
            if p0 is not None:
                p0 = np.concatenate((p0, new_pts), axis=0)
            else:
                p0 = new_pts

        mask_draw = np.zeros_like(frame)

    # =========================
    # Dibujar ROI (visual)
    # =========================
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # =========================
    # Visualización
    # =========================
    output = cv2.add(frame, mask_draw)

    cv2.putText(output, f"Puntos activos: {0 if p0 is None else len(p0)}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.imshow("Optical Flow (Robusto)", output)
    cv2.imshow("Mascara ROI", corner_mask)

    # =========================
    # Teclas
    # =========================
    key = cv2.waitKey(30) & 0xFF

    if key == ord('s'):
        cv2.imwrite("captura_tracking.png", output)
        cv2.imwrite("mascara.png", corner_mask)
        print("Capturas guardadas")

    if key == ord('q'):
        break

    # =========================
    # Actualización
    # =========================
    old_gray = frame_gray.copy()

video.release()
cv2.destroyAllWindows()
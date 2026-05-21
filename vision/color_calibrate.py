import cv2
import numpy as np

def nothing(x):
    pass

# Inicializar ventana gráfica única
cv2.namedWindow("Calibracion HSV")

# Crear barras deslizantes tradicionales (0 a 179 para Hue)
cv2.createTrackbar("H Min", "Calibracion HSV", 0, 179, nothing)
cv2.createTrackbar("H Max", "Calibracion HSV", 179, 179, nothing)
cv2.createTrackbar("S Min", "Calibracion HSV", 0, 255, nothing)
cv2.createTrackbar("S Max", "Calibracion HSV", 255, 255, nothing)
cv2.createTrackbar("V Min", "Calibracion HSV", 0, 255, nothing)
cv2.createTrackbar("V Max", "Calibracion HSV", 255, 255, nothing)

# Iniciar la cámara web de la ASUS TUF
cap = cv2.VideoCapture(0)

print("--- Calibrador Tradicional de Alta Precisión Iniciado ---")
print("Ajusta las barras hasta que el objeto quede completamente BLANCO")
print("y el fondo completamente NEGRO. Presiona 'q' para guardar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.resize(frame, (500, 375))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Captura manual estricta de los sliders
    h_min = cv2.getTrackbarPos("H Min", "Calibracion HSV")
    h_max = cv2.getTrackbarPos("H Max", "Calibracion HSV")
    s_min = cv2.getTrackbarPos("S Min", "Calibracion HSV")
    s_max = cv2.getTrackbarPos("S Max", "Calibracion HSV")
    v_min = cv2.getTrackbarPos("V Min", "Calibracion HSV")
    v_max = cv2.getTrackbarPos("V Max", "Calibracion HSV")
    
    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])
    
    # Única máscara directa (Rango continuo puro)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Filtros idénticos al detector oficial para garantizar precisión matemática
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Convertir la máscara a 3 canales para poder pegarla al frame original
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Vista lado a lado: Izquierda (Real) | Derecha (Máscara Binaria Pura)
    horizontal_concat = np.hstack((frame, mask_bgr))
    
    cv2.imshow("Calibracion HSV", horizontal_concat)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Impresión limpia para copiar directo a tu vision.py
print("\n=== ¡VALORES CALIBRADOS EXITOSAMENTE! ===")
print(f"lower_bound = np.array([{h_min}, {s_min}, {v_min}])")
print(f"upper_bound = np.array([{h_max}, {s_max}, {v_max}])")

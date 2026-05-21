import cv2
import numpy as np

def detect_object(frame):
    """
    P6 / ROL 4: Procesamiento multiobjetivo unificado con calibración ASUS TUF.
    Filtra por Región de Interés (ROI) para ignorar el rostro y clasifica el color.
    Retorna: (x_mm, y_mm, "NOMBRE_DEL_COLOR") o None
    """
    if frame is None:
        return None

    # =================================================================
    # 🛡️ ESCUDO ANTI-ROSTRO (Región de Interés)
    # =================================================================
    # Volvemos invisible tu cara tapando los primeros 180 píxeles de alto.
    frame_proc = frame.copy()
    frame_proc[0:180, :] = 0 

    hsv = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)
    
    masks = {}

    # =================================================================
    # 🎨 INTEGRACIÓN DE TU NUEVA CALIBRACIÓN REAL
    # =================================================================
    
    # 🟡 Amarillo (Tus nuevos valores: [17, 51, 168] a [56, 255, 255])
    lower_yellow = np.array([17, 51, 168])
    upper_yellow = np.array([56, 255, 255])
    masks["AMARILLO"] = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 🔵 Azul (Tus nuevos valores: [94, 89, 0] a [132, 255, 240])
    lower_blue = np.array([94, 89, 0])
    upper_blue = np.array([132, 255, 240])
    masks["AZUL"] = cv2.inRange(hsv, lower_blue, upper_blue)

    # 🟢 Verde (Tu calibración base acotada para que no invada otros colores)
    lower_green = np.array([56, 112, 76])
    upper_green = np.array([90, 255, 215])  # Acotado a 90 para bloquear falsos positivos
    masks["VERDE"] = cv2.inRange(hsv, lower_green, upper_green)

    # 🔴 Rojo (Tu calibración aplicada a los extremos reales del Matiz)
    # Evita que el Hue 0-179 se trague toda la pantalla y tu rostro.
    lower_red1 = np.array([0, 116, 185])
    upper_red1 = np.array([15, 255, 255])
    mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
    
    lower_red2 = np.array([165, 116, 185])
    upper_red2 = np.array([179, 255, 255])
    mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
    masks["ROJO"] = cv2.bitwise_or(mask_r1, mask_r2)

    # =================================================================
    # 🔍 EVALUACIÓN DE CONTORNOS Y PRIORIDAD
    # =================================================================
    best_area = 0
    best_contour = None
    color_detectado = "NINGUNO"

    for name, mask in masks.items():
        # Limpieza morfológica idéntica a tus scripts de los .zip
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            
            # Filtro de área mínimo para ignorar ruidos pequeños en la mesa
            if area > 400 and area > best_area:
                best_area = area
                best_contour = largest
                color_detectado = name

    if best_contour is None:
        return None

    # =================================================================
    # 🎯 COORDENADAS GEOMÉTRICAS (Cálculo original de tus scripts)
    # =================================================================
    M = cv2.moments(best_contour)
    if M["m00"] == 0:
        return None
        
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    # Conversión píxel -> mm usando tu constante de escala (0.55)
    centro_camera_x = frame.shape[1] // 2
    centro_camera_y = frame.shape[0] // 2
    escala_mm_pixel = 0.55  
    
    x_mm = (cx - centro_camera_x) * escala_mm_pixel
    y_mm = (centro_camera_y - cy) * escala_mm_pixel
    
    # Dibujamos en el frame original que tú estás viendo en la pantalla
    cv2.circle(frame, (cx, cy), 7, (0, 255, 0), -1)
    # Línea roja de guía para saber dónde termina la zona muerta anti-rostro
    cv2.line(frame, (0, 180), (frame.shape[1], 180), (0, 0, 255), 1)
    
    return (float(x_mm), float(y_mm), color_detectado)

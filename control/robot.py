from pymycobot.mycobot import MyCobot
from datetime import datetime
import csv
import time

mc = MyCobot('/dev/ttyUSB0', 1000000)
time.sleep(0.5)

SPEED          = 30
GRIPPER_SPEED  = 50
Z_APROXIMACION = 125   # ← actualizado con tu diagnóstico
Z_AGARRE       = 140.0  # ← actualizado con tu diagnóstico

watch_pose = [15.82, -0.26, -13.97, -62.57, 4.65, -29.35]

POSES_DESCARGA = {
    "AZUL": {
        "lift_pose":     [119.09, -18.54, -57.21, -10.81,  2.46, -14.50],
        "place_pose":    [115.31, -46.66, -60.02,   7.11,  0.79, -26.10],
        "angulo_prueba": [118.38, -19.68, -84.81,  23.20,  2.98, -10.37],
    },
    "VERDE": {
        "lift_pose":     [101.16,  -8.70, -68.46,  -6.41,  1.93, -33.31],
        "place_pose":    [101.60, -46.84, -57.48,   0.61,  2.02, -33.31],
        "angulo_prueba": [100.63, -17.84, -58.35,  -0.35,  1.93, -35.06],
    },
    "ROJO": {
        "lift_pose":     [ 84.46, -22.85, -58.35,  -0.61,  4.21, -49.65],
        "place_pose":    [ 82.79, -48.95, -58.35,  10.63,  3.33, -49.65],
        "angulo_prueba": [ 83.40, -15.90, -58.88,   9.49,  3.33, -48.69],
    },
    "AMARILLO": {
        "lift_pose":     [ 68.46, -25.04, -46.58,  -5.27,  1.66, -65.91],
        "place_pose":    [ 68.64, -57.56, -46.05,  13.00,  1.31, -65.91],
        "angulo_prueba": None,
    },
}
LOG_FILE = f"log_control_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv" 
def _init_log():   
    with open(LOG_FILE, "w", newline="") as f: 
        w = csv.writer(f)       
        w.writerow(["ciclo", "timestamp", "color","dx_mm", "dy_mm","fase", "resultado", "duracion_s", "nota"]) 
def _log(ciclo, color, dx_mm, dy_mm, fase, resultado, duracion_s, nota=""):   
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")   
    with open(LOG_FILE, "a", newline="") as f:       
        w = csv.writer(f)       
        w.writerow([ciclo, ts, color, dx_mm, dy_mm, fase, resultado, round(duracion_s, 2), nota])   
        estado = "" if resultado == "OK" else ""   
        print(f" [{ciclo}] {estado} {fase:<20} | {duracion_s:.1f}s | {nota}")
def get_coords_robusto(intentos=10, espera=0.6):   
    for i in range(intentos):       
        coords = mc.get_coords()       
        if coords and len(coords) == 6 and any(v != 0 for v in coords):           
            print(f"  Coords (intento {i+1}): {[round(v,2) for v in coords]}")            r
            return coords       
        print(f"  Intento {i+1}: {coords}")       
        time.sleep(espera)   
    raise RuntimeError("No se obtuvieron coords válidas tras reintentos.")      
def open_gripper():
    mc.set_gripper_value(97, GRIPPER_SPEED)
    time.sleep(1.5)

def close_gripper():
    mc.set_gripper_value(20, GRIPPER_SPEED)
    time.sleep(1.5)

def get_coords_robusto(intentos=10, espera=0.6):
    for i in range(intentos):
        coords = mc.get_coords()
        if coords and len(coords) == 6 and any(v != 0 for v in coords):
            print(f" Coords intento {i+1}: {[round(v,2) for v in coords]}")
            return coords
        print(f" Intento {i+1}: {coords}")
        time.sleep(espera)
    raise RuntimeError(" No se obtuvieron coords válidas.")

# ── Verificar detección ──
if not deteccion["activo"]:
    print("  Cámara aún no confirmó objeto.")
    print("    Espera  LISTO y corre esta celda de nuevo.")
else:
    color = deteccion["color"]
    dx_mm = deteccion["dx_mm"]
    dy_mm = deteccion["dy_mm"]
    poses = POSES_DESCARGA[color]

    print(f" Color: {color} | dx={dx_mm:+.1f}mm  dy={dy_mm:+.1f}mm")

    # 1. Watch pose
    print(" Yendo a watch_pose...")
    mc.send_angles(watch_pose, SPEED)
    time.sleep(4)

    # 2. Coords reales
    print(" Leyendo coordenadas reales...")
    X1, Y1, Z1, Rx, Ry, Rz = get_coords_robusto()

    # 3. Suma dinámica
    X_bloque = round(197.3 + dy_mm, 2)
    Y_bloque = round(2.9 - dx_mm, 2)
    #print(f" X: {X1:.1f} + {dx_mm:.1f} = {X_bloque}")
    #print(f" Y: {Y1:.1f} + {dy_mm:.1f} = {Y_bloque}")

    #X_bloque = 197.3
    #Y_bloque = 2.9
    open_gripper()

    # 4. Sobre el bloque
    print("↕ Aproximación...")
    mc.send_coords([X_bloque, Y_bloque, 125, Rx, Ry, Rz], SPEED, 1)
    time.sleep(3)

    # 5. Bajar
    print("⬇ Bajando...")
    mc.send_coords([X_bloque, Y_bloque, 125, Rx, Ry, Rz], SPEED, 1)
    time.sleep(3)

    # 6. Agarrar
    print("Cerrando gripper...")
    close_gripper()

    # 7. Subir
    print("⬆ Subiendo...")
    mc.send_coords([X_bloque, Y_bloque, 125, Rx, Ry, Rz], SPEED, 1)
    time.sleep(3)

    # 8. Lift pose
    print(f" Lift → {color}...")
    mc.send_angles(poses["lift_pose"], SPEED)
    time.sleep(2.5)

    # 9. Place pose
    print(f" Place → {color}...")
    mc.send_angles(poses["place_pose"], SPEED)
    time.sleep(2.5)

    # 10. Soltar
    print(" Soltando...")
    open_gripper()

    # 11. Angulo prueba
    if poses["angulo_prueba"]:
        print("↩ Ángulo prueba...")
        mc.send_angles(poses["angulo_prueba"], SPEED)
        time.sleep(2.5)

    # 12. Watch pose
    print(" Volviendo a watch_pose...")
    mc.send_angles(watch_pose, SPEED)
    time.sleep(3)

    print(f" Listo — {color} depositado.")
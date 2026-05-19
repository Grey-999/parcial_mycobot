from vision.detector import detectar_objeto
from cinematica.ik import calcular_ik
from control.robot import (
    mover_robot,
    agarrar_objeto,
    depositar_objeto
)

import logging
import time

# =========================
# LOGGING
# =========================

logging.basicConfig(
    filename="logs/sistema.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# =========================
# ESTADO INICIAL
# =========================

estado = "IDLE"

print("Sistema iniciado")

logging.info("Sistema iniciado")

# =========================
# LOOP PRINCIPAL
# =========================

for ciclo in range(5):

    print(f"\n--- CICLO {ciclo + 1} ---")

    # ---------------------
    # DETECTAR
    # ---------------------

    estado = "DETECTANDO"

    print(estado)

    logging.info(estado)

    objeto = detectar_objeto()

    if objeto is None:

        print("No se detectó objeto")

        logging.info("No se detectó objeto")

        continue

    x, y, z = objeto

    print(f"Objeto detectado: {objeto}")

    logging.info(f"Objeto detectado: {objeto}")

    # ---------------------
    # IK
    # ---------------------

    estado = "CALC_IK"

    print(estado)

    logging.info(estado)

    angles = calcular_ik(x, y, z)

    print(f"Ángulos calculados: {angles}")

    logging.info(f"Ángulos calculados: {angles}")

    # ---------------------
    # AGARRAR
    # ---------------------

    estado = "AGARRANDO"

    print(estado)

    logging.info(estado)

    # mover_robot(mc, angles)
    # agarrar_objeto(mc)

    print("Objeto agarrado")

    logging.info("Objeto agarrado")

    # ---------------------
    # DEPOSITAR
    # ---------------------

    estado = "DEPOSITAR"

    print(estado)

    logging.info(estado)

    # depositar_objeto(mc)

    print("Objeto depositado")

    logging.info("Objeto depositado")

    logging.info("Ciclo completado")

    time.sleep(2)

estado = "IDLE"

print("Sistema finalizado")

logging.info("Sistema finalizado")
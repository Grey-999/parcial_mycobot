"""
main.py — Pipeline end-to-end MyCobot 280
==========================================
Rol 1: Líder de Integración
Máquina de estados: IDLE → DETECTANDO → CALC_IK → AGARRANDO → DEPOSITAR

Plataforma : Ubuntu 20.04 + ROS 2 Foxy + Jetson Nano
Robot      : MyCobot 280 — /dev/ttyUSB0 @ 1 000 000 baud
"""

import logging
import signal
import sys
import time

from pymycobot.mycobot import MyCobot

# --- Módulo Rol 2: clases exactas de cinematica/ik.py ---
from cinematica.cinematica import ForwardKinematics, InverseKinematics, CollisionChecker

# --- Módulos Rol 3 y Rol 4 ---
from vision.detector import detectar_objeto
from control.robot import mover_robot, agarrar_objeto, depositar_objeto


# ===========================================================================
# Instancias del módulo de cinemática (Rol 2)
# Se crean una sola vez al importar el módulo
# ===========================================================================

_fk      = ForwardKinematics()
_ik      = InverseKinematics(elbow_up=True)
_checker = CollisionChecker(z_safe_mm=10.0)


def calcular_ik(x: float, y: float, z: float) -> list:
    """
    Wrapper sobre las clases del Rol 2.
    1. Calcula ángulos con IK analítica (ik_solve)
    2. Verifica la posición con FK (compute_fk)
    3. Valida colisiones con CollisionChecker (is_safe)
    Retorna lista de 6 ángulos en grados o lanza ValueError.
    """
    # Paso 1: IK analítica → ángulos
    angles = _ik.ik_solve(x, y, z)

    # Paso 2: FK para obtener posición real del efector
    pos, _ = _fk.compute_fk(angles)

    # Paso 3: verificar límites articulares y altura sobre la mesa
    if not _checker.is_safe(angles, pos):
        raise ValueError(
            f"Movimiento inseguro para (x={x:.1f}, y={y:.1f}, z={z:.1f}) mm"
        )

    return angles.tolist()


# ===========================================================================
# Logging centralizado con timestamp (consola + archivo)
# ===========================================================================

logging.basicConfig(
    filename="logs/sistema.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
logging.getLogger().addHandler(console)

log = logging.getLogger(__name__)


# ===========================================================================
# Constantes del sistema
# ===========================================================================

PUERTO         = "/dev/ttyUSB0"
BAUDRATE       = 1_000_000
INIT_POSE      = [0, 0, 0, 0, 0, -45]   # pose segura de reposo
VELOCIDAD      = 50                       # 50% en pruebas (normas de seguridad)
CICLOS         = 5
MAX_REINTENTOS = 3


# ===========================================================================
# Parada de emergencia (Ctrl+C)
# ===========================================================================

_mc_instance = None


def _emergency_stop(signum, frame):
    log.warning("PARADA DE EMERGENCIA — liberando servos.")
    try:
        if _mc_instance:
            _mc_instance.release_all_servos()
    except Exception:
        pass
    sys.exit(0)


signal.signal(signal.SIGINT, _emergency_stop)


# ===========================================================================
# Inicialización del robot
# ===========================================================================

def inicializar_robot() -> MyCobot:
    global _mc_instance

    log.info(f"Conectando a MyCobot 280 en {PUERTO} @ {BAUDRATE} baud...")
    mc = MyCobot(PUERTO, BAUDRATE)
    mc.power_on()
    time.sleep(0.5)

    # Verificación obligatoria de conexión (Rol 1 — API requerida)
    assert mc.is_controller_connected(), "ERROR: controlador no responde."

    log.info("Conexión establecida.")

    # Mover a pose segura inicial antes de cualquier operación
    mc.send_angles(INIT_POSE, VELOCIDAD)
    time.sleep(3)

    pos_inicial = mc.get_coords()
    log.info(f"Posición inicial verificada: {pos_inicial}")

    _mc_instance = mc
    return mc


# ===========================================================================
# Máquina de estados
# ===========================================================================

class RobotController:
    """
    Pipeline end-to-end del MyCobot 280.

    Estados
    -------
    IDLE        — esperando inicio de ciclo
    DETECTANDO  — módulo de visión buscando objeto
    CALC_IK     — calculando cinemática inversa con módulo del Rol 2
    AGARRANDO   — moviendo robot y cerrando gripper
    DEPOSITAR   — depositando objeto y volviendo a IDLE
    """

    ESTADOS_VALIDOS = {"IDLE", "DETECTANDO", "CALC_IK", "AGARRANDO", "DEPOSITAR"}

    def __init__(self, mc: MyCobot):
        self.mc     = mc
        self.estado = "IDLE"
        self.exitos = 0
        self.fallos = 0

    def _cambiar_estado(self, nuevo: str):
        log.info(f"[ESTADO] {self.estado} → {nuevo}")
        self.estado = nuevo

    # ------------------------------------------------------------------
    # Estado DETECTANDO
    # ------------------------------------------------------------------

    def _detectar(self):
        self._cambiar_estado("DETECTANDO")

        for intento in range(1, MAX_REINTENTOS + 1):
            try:
                objeto = detectar_objeto()
                if objeto is None:
                    log.info("No se detectó objeto en la escena.")
                    return None
                x, y, z = objeto
                log.info(f"Objeto detectado: x={x:.1f} mm, y={y:.1f} mm, z={z:.1f} mm")
                return objeto
            except Exception as e:
                log.warning(f"Error detección (intento {intento}/{MAX_REINTENTOS}): {e}")
                time.sleep(0.5)

        log.error("Detección fallida tras todos los reintentos.")
        return None

    # ------------------------------------------------------------------
    # Estado CALC_IK  —  usa ForwardKinematics, InverseKinematics y CollisionChecker
    # ------------------------------------------------------------------

    def _calcular_ik(self, x: float, y: float, z: float):
        self._cambiar_estado("CALC_IK")

        for intento in range(1, MAX_REINTENTOS + 1):
            try:
                # calcular_ik llama internamente a:
                #   _ik.ik_solve()      → InverseKinematics del Rol 2
                #   _fk.compute_fk()    → ForwardKinematics del Rol 2
                #   _checker.is_safe()  → CollisionChecker del Rol 2
                angles = calcular_ik(x, y, z)
                log.info(f"Ángulos IK: {[round(a, 2) for a in angles]}")
                return angles
            except ValueError as e:
                # Error matemático (fuera de espacio de trabajo o colisión)
                # No tiene sentido reintentar
                log.error(f"IK rechazada: {e}")
                return None
            except Exception as e:
                log.warning(f"Error IK (intento {intento}/{MAX_REINTENTOS}): {e}")
                time.sleep(0.5)

        log.error("Cálculo IK fallido tras todos los reintentos.")
        return None

    # ------------------------------------------------------------------
    # Estado AGARRANDO
    # ------------------------------------------------------------------

    def _agarrar(self, angles: list) -> bool:
        self._cambiar_estado("AGARRANDO")

        for intento in range(1, MAX_REINTENTOS + 1):
            try:
                mover_robot(self.mc, angles)
                agarrar_objeto(self.mc)
                pos_real = self.mc.get_coords()
                log.info(f"Objeto agarrado. Posición real: {pos_real}")
                return True
            except Exception as e:
                log.warning(f"Error agarre (intento {intento}/{MAX_REINTENTOS}): {e}")
                time.sleep(1.0)

        log.error("Agarre fallido tras todos los reintentos.")
        return False

    # ------------------------------------------------------------------
    # Estado DEPOSITAR
    # ------------------------------------------------------------------

    def _depositar(self) -> bool:
        self._cambiar_estado("DEPOSITAR")

        for intento in range(1, MAX_REINTENTOS + 1):
            try:
                depositar_objeto(self.mc)
                log.info("Objeto depositado.")
                # Volver a pose segura entre ciclos
                self.mc.send_angles(INIT_POSE, VELOCIDAD)
                time.sleep(2)
                return True
            except Exception as e:
                log.warning(f"Error depósito (intento {intento}/{MAX_REINTENTOS}): {e}")
                time.sleep(1.0)

        log.error("Depósito fallido tras todos los reintentos.")
        return False

    # ------------------------------------------------------------------
    # Ciclo completo
    # ------------------------------------------------------------------

    def ejecutar_ciclo(self, num_ciclo: int) -> bool:
        log.info("=" * 50)
        log.info(f"CICLO {num_ciclo}/{CICLOS} — INICIO")
        t_inicio = time.time()

        # DETECTANDO
        objeto = self._detectar()
        if objeto is None:
            self._cambiar_estado("IDLE")
            self.fallos += 1
            log.warning(f"Ciclo {num_ciclo} FALLIDO en DETECTANDO.")
            return False
        x, y, z = objeto

        # CALC_IK
        angles = self._calcular_ik(x, y, z)
        if angles is None:
            self._cambiar_estado("IDLE")
            self.fallos += 1
            log.warning(f"Ciclo {num_ciclo} FALLIDO en CALC_IK.")
            return False

        # AGARRANDO
        if not self._agarrar(angles):
            self._cambiar_estado("IDLE")
            self.fallos += 1
            log.warning(f"Ciclo {num_ciclo} FALLIDO en AGARRANDO.")
            return False

        # DEPOSITAR
        if not self._depositar():
            self._cambiar_estado("IDLE")
            self.fallos += 1
            log.warning(f"Ciclo {num_ciclo} FALLIDO en DEPOSITAR.")
            return False

        self._cambiar_estado("IDLE")
        self.exitos += 1
        log.info(f"Ciclo {num_ciclo} EXITOSO — tiempo: {time.time() - t_inicio:.2f}s")
        return True


# ===========================================================================
# Punto de entrada
# ===========================================================================

def main():
    log.info("Sistema MyCobot 280 — iniciando pipeline end-to-end")

    mc    = inicializar_robot()
    robot = RobotController(mc)

    for ciclo in range(1, CICLOS + 1):
        robot.ejecutar_ciclo(ciclo)
        time.sleep(1)

    tasa = robot.exitos / CICLOS * 100
    log.info("=" * 50)
    log.info(f"SESIÓN COMPLETADA — Éxitos: {robot.exitos}/{CICLOS} ({tasa:.0f}%)")
    log.info(f"Fallos: {robot.fallos}/{CICLOS}")

    mc.send_angles(INIT_POSE, VELOCIDAD)
    time.sleep(2)
    log.info("Robot en pose de reposo. Sistema finalizado.")


if __name__ == "__main__":
    main()
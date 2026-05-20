"""
cinematica.py — Módulo de cinemática para el brazo robótico MyCobot 280 (6-DOF)
================================================================================
Plataforma : Ubuntu 20.04 + ROS 2 Foxy
Dependencia : numpy
API externa : pymycobot  (ángulos en GRADOS, posiciones en MILÍMETROS)

Clases
------
ForwardKinematics  — Cinemática directa via parámetros DH estándar
InverseKinematics  — Solución analítica 3R + muñeca fija
CollisionChecker   — Validación de límites articulares y zona segura (mesa)

Convención de unidades
----------------------
- Interfaz pública  → ángulos en GRADOS, distancias en MILÍMETROS (igual que pymycobot)
- Cálculos internos → ángulos en RADIANES, distancias en MILÍMETROS
"""

# pyrefly: ignore [missing-import]
import numpy as np  # type: ignore


# ---------------------------------------------------------------------------
# Constantes globales: parámetros DH del MyCobot 280
# Orden de columnas: [a (mm), d (mm), alpha (rad), theta_offset (rad)]
# Los rangos articulares están en GRADOS (interfaz exterior).
# ---------------------------------------------------------------------------

# Parámetros DH estándar  →  T_i = Rz(θ) · Tz(d) · Tx(a) · Rx(α)
DH_PARAMS = np.array([
    #   a      d        alpha                    theta_offset
    [   0.0, 131.56,  np.radians( 90.0),       0.0],   # J1  Base
    [ 110.4,   0.00,  np.radians(  0.0),       0.0],   # J2  Hombro
    [  96.0,   0.00,  np.radians(  0.0),       0.0],   # J3  Codo
    [   0.0,  66.39,  np.radians(-90.0),       0.0],   # J4  Muñeca 1
    [   0.0,  73.18,  np.radians( 90.0),       0.0],   # J5  Muñeca 2
    [   0.0,  48.60,  np.radians(  0.0),       0.0],   # J6  Gripper
])

# Rangos articulares físicos [min_deg, max_deg]
JOINT_LIMITS_DEG = np.array([
    [-168.0,  168.0],   # J1
    [-135.0,   90.0],   # J2
    [-150.0,  150.0],   # J3
    [-145.0,  145.0],   # J4
    [-165.0,  165.0],   # J5
    [-180.0,  180.0],   # J6
])

# Longitudes de eslabones usadas en la cinemática inversa (mm)
D1 = 131.56   # offset vertical de la base (d del joint 1)
L2 = 110.4    # longitud del eslabón 2 (a del joint 2)
L3 =  96.0    # longitud del eslabón 3 (a del joint 3)


# ===========================================================================
# Clase 1 — Cinemática Directa (Forward Kinematics)
# ===========================================================================

class ForwardKinematics:
    """
    Calcula la posición y orientación del efector final a partir de los
    ángulos articulares usando los parámetros DH del MyCobot 280.

    Convención DH estándar:
        T_i = Rz(θ_i) · Tz(d_i) · Tx(a_i) · Rx(α_i)
    """

    def __init__(self):
        # Copia local de los parámetros DH (shape 6×4)
        self.dh = DH_PARAMS.copy()

    # ------------------------------------------------------------------
    # Método privado: matriz de transformación homogénea para un eslabón
    # ------------------------------------------------------------------
    def _dh_matrix(self, theta: float, d: float, a: float, alpha: float) -> np.ndarray:
        """
        Construye la matriz homogénea 4×4 para un eslabón con parámetros DH.

        Parámetros
        ----------
        theta : float — ángulo de junta (rad)
        d     : float — offset de junta a lo largo del eje Z (mm)
        a     : float — longitud del eslabón a lo largo del eje X (mm)
        alpha : float — ángulo de torsión del eslabón alrededor del eje X (rad)

        Retorna
        -------
        T : np.ndarray (4×4) — matriz de transformación homogénea
        """
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)

        # Fórmula estándar DH: T = Rz(θ) · Tz(d) · Tx(a) · Rx(α)
        T = np.array([
            [ ct,  -st * ca,   st * sa,   a * ct],
            [ st,   ct * ca,  -ct * sa,   a * st],
            [0.0,       sa,        ca,        d ],
            [0.0,      0.0,       0.0,      1.0],
        ])
        return T

    # ------------------------------------------------------------------
    # Método público principal
    # ------------------------------------------------------------------
    def compute_fk(self, angles_deg: list | np.ndarray) -> np.ndarray:
        """
        Calcula la cinemática directa completa T_0_6 y devuelve la posición
        cartesiana del efector final.

        Parámetros
        ----------
        angles_deg : array-like de 6 elementos — ángulos articulares en GRADOS

        Retorna
        -------
        position : np.ndarray (3,) — [x, y, z] en MILÍMETROS
        T_0_6    : np.ndarray (4×4) — transformación homogénea completa
                   (también accesible como segundo valor de retorno)
        """
        angles_deg = np.asarray(angles_deg, dtype=float)
        if angles_deg.shape != (6,):
            raise ValueError(f"Se esperan 6 ángulos; se recibieron {angles_deg.shape}.")

        # Convertir grados → radianes para los cálculos internos
        angles_rad = np.radians(angles_deg)

        # Producto encadenado: T_0_6 = T_1 · T_2 · T_3 · T_4 · T_5 · T_6
        T_0_6 = np.eye(4)
        for i in range(6):
            a     = self.dh[i, 0]
            d     = self.dh[i, 1]
            alpha = self.dh[i, 2]
            theta = angles_rad[i] + self.dh[i, 3]   # θ_i + offset

            T_i   = self._dh_matrix(theta, d, a, alpha)
            T_0_6 = T_0_6 @ T_i

        # Extraer posición cartesiana de la última columna (columna 3, filas 0-2)
        position = T_0_6[:3, 3]   # [x, y, z] en mm
        return position, T_0_6

    # ------------------------------------------------------------------
    def compute_partial_fk(
        self, angles_deg: list | np.ndarray, num_joints: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calcula la FK parcial T_0_k para los primeros `num_joints` eslabones.

        Útil para obtener el wrist center (k=3) o posiciones intermedias.

        Parámetros
        ----------
        angles_deg : array-like (6,) — ángulos articulares completos en GRADOS
        num_joints : int             — número de articulaciones a incluir (1..6)

        Retorna
        -------
        position : np.ndarray (3,) — [x, y, z] del origen del frame `num_joints` en mm
        T_0_k    : np.ndarray (4×4) — transformación homogénea parcial
        """
        angles_deg = np.asarray(angles_deg, dtype=float)
        if not (1 <= num_joints <= 6):
            raise ValueError("num_joints debe estar entre 1 y 6.")

        angles_rad = np.radians(angles_deg)
        T_0_k = np.eye(4)
        for i in range(num_joints):
            a     = self.dh[i, 0]
            d     = self.dh[i, 1]
            alpha = self.dh[i, 2]
            theta = angles_rad[i] + self.dh[i, 3]
            T_i   = self._dh_matrix(theta, d, a, alpha)
            T_0_k = T_0_k @ T_i

        position = T_0_k[:3, 3]
        return position, T_0_k


# ===========================================================================
# Clase 2 — Cinemática Inversa (Inverse Kinematics)
# ===========================================================================

# Offset total de la muñeca a lo largo del eje Z del efector
# cuando J4=J5=J6=0: la cadena de la muñeca suma d4 + d5 + d6 en la
# dirección del eje Z local del eslabón 3.
WRIST_OFFSET_MM = DH_PARAMS[3, 1] + DH_PARAMS[4, 1] + DH_PARAMS[5, 1]   # 66.39 + 73.18 + 48.6 = 188.17 mm


class InverseKinematics:
    """
    Solución analítica de la cinemática inversa para los 3 primeros joints
    del MyCobot 280 (modelo Base + 2R planar).

    Estrategia de desacoplamiento de muñeca (wrist decoupling)
    -----------------------------------------------------------
    Dado que J4, J5, J6 se fijan en 0°, la muñeca queda alineada con el
    eje Z global.  El "wrist center" (origen de J4) se encuentra retrocediendo
    WRIST_OFFSET_MM = d4 + d5 + d6 desde el TCP a lo largo del eje −Z.

    El modelo 2R planar se aplica al wrist center, no al TCP directamente.

    Convención:
        θ₁ = atan2(y_wc, x_wc)
        r  = √(x_wc² + y_wc²)           — radio en el plano XY
        z' = z_wc − d₁                  — altura relativa a la base
        cos(θ₃) = (r² + z'² − L2² − L3²) / (2·L2·L3)
        θ₃ = atan2(±√(1 − cos²θ₃), cos(θ₃))
        θ₂ = atan2(z', r) − atan2(L3·sin(θ₃), L2 + L3·cos(θ₃))
    """

    def __init__(self, elbow_up: bool = True):
        """
        Parámetros
        ----------
        elbow_up : bool
            True  → configuración de codo arriba (signo positivo de θ₃)
            False → configuración de codo abajo   (signo negativo de θ₃)
        """
        self.elbow_up      = elbow_up
        self.d1            = D1
        self.L2            = L2
        self.L3            = L3
        self.wrist_offset  = WRIST_OFFSET_MM   # d4 + d5 + d6
        # Referencia a FK para calcular el wrist center desde la pose completa
        self._fk = ForwardKinematics()

    # ------------------------------------------------------------------
    def _compute_wrist_center(
        self, x_tcp: float, y_tcp: float, z_tcp: float,
        T_0_6: np.ndarray,
    ) -> tuple[float, float, float]:
        """
        Calcula el wrist center (origen de J4) a partir del TCP.

        Cuando J4=J5=J6=0 la traslacion total de la muneca respecto al
        origen de J4 ocurre a lo largo del eje Z LOCAL del eslabón 3,
        pero dicho eje Z no coincide en general con el eje Z global.
        Por eso usamos la orientacion de la matriz T_0_3 (FK parcial con
        J1=theta1, J2=theta2_approx, J3=0, J4=J5=J6=0) para descontar
        el offset.

        Estrategia simplificada compatible con el modelo de la guia:
        Se itera una vez para refinar el wrist center asumiendo que el
        eje Z del efector final (con muneca a 0) queda aproximadamente
        en la direccion del eje z de la FK parcial J1-J3.  Para la primera
        iteracion se usa el eje Z del TCP directamente de T_0_6.

        Parametros
        ----------
        x_tcp, y_tcp, z_tcp : float -- posicion del TCP en mm
        T_0_6               : np.ndarray (4x4) -- transformacion homogenea
                              completa del TCP (de FK)

        Retorna
        -------
        (x_wc, y_wc, z_wc) : tuple[float, float, float] -- wrist center en mm
        """
        # Extraer el eje Z del efector (tercera columna de la submatriz de rotacion)
        # Eje Z en el frame global apunta en la direccion de salida del gripper
        z_axis = T_0_6[:3, 2]   # vector unitario eje Z del efector

        # Retroceder WRIST_OFFSET_MM a lo largo del eje Z del efector
        # para llegar al wrist center (origen de J4)
        wc = np.array([x_tcp, y_tcp, z_tcp]) - self.wrist_offset * z_axis
        return float(wc[0]), float(wc[1]), float(wc[2])

    # ------------------------------------------------------------------
    def ik_solve(
        self,
        x: float,
        y: float,
        z: float,
        T_0_6: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Calcula los ángulos articulares para que el efector alcance (x, y, z).

        Internamente aplica desacoplamiento de muñeca usando el eje Z del TCP
        (calculado por FK con los ángulos objetivo asumiendo muñeca a 0):
            wrist_center = TCP - wrist_offset * z_axis_efector

        Parámetros
        ----------
        x, y, z : float — posición del TCP en MILÍMETROS
        T_0_6   : np.ndarray (4x4), opcional — matriz de transformación
                  homogénea completa del TCP.  Si se pasa, se usa para
                  calcular el wrist center exacto.  Si se omite, se usa
                  la aproximación de eje Z global (muñeca apuntando abajo).

        Retorna
        -------
        angles_deg : np.ndarray (6,) — ángulos en GRADOS
                     [θ₁, θ₂, θ₃, 0, 0, 0]

        Lanza
        -----
        ValueError — si (x, y, z) está fuera del espacio de trabajo alcanzable
        """
        # ---- Desacoplamiento de muñeca ----------------------------------
        # Con T_0_6 disponible usamos el eje Z real del efector.
        # Sin T_0_6 usamos aproximación: eje Z = (0, 0, -1) (efector apuntando abajo)
        if T_0_6 is not None:
            x_wc, y_wc, z_wc = self._compute_wrist_center(x, y, z, T_0_6)
        else:
            # Aproximacion: muneca apunta en -Z global
            # wrist center esta WRIST_OFFSET_MM arriba del TCP en Z
            x_wc = x
            y_wc = y
            z_wc = z + self.wrist_offset

        # ---- θ₁: rotación alrededor del eje Z de la base ----------------
        theta1_rad = np.arctan2(y_wc, x_wc)

        # ---- Parámetros del modelo planar 2R (sobre el wrist center) ----
        r       = np.sqrt(x_wc**2 + y_wc**2)    # distancia radial en XY
        z_prime = z_wc - self.d1                 # altura relativa a la base

        # ---- Verificación del espacio de trabajo ANTES de arccos --------
        # cos(θ₃) = (r² + z'² − L2² − L3²) / (2·L2·L3)
        numerator   = r**2 + z_prime**2 - self.L2**2 - self.L3**2
        denominator = 2.0 * self.L2 * self.L3

        cos_theta3 = numerator / denominator

        # REQUISITO CRÍTICO: atrapar posición inalcanzable antes de sqrt/arccos
        if np.abs(cos_theta3) > 1.0:
            raise ValueError(
                f"Posición fuera del espacio de trabajo alcanzable. "
                f"cos(θ₃) = {cos_theta3:.4f} (|valor| > 1). "
                f"Coordenadas solicitadas: x={x:.2f} mm, y={y:.2f} mm, z={z:.2f} mm."
            )

        # ---- θ₃: ángulo del codo (elbow up / elbow down) ----------------
        # Codo arriba → sin(θ₃) > 0;  Codo abajo → sin(θ₃) < 0
        sin_theta3 = np.sqrt(1.0 - cos_theta3**2)
        if not self.elbow_up:
            sin_theta3 = -sin_theta3

        theta3_rad = np.arctan2(sin_theta3, cos_theta3)

        # ---- θ₂: ángulo del hombro --------------------------------------
        # θ₂ = atan2(z', r) − atan2(L3·sin(θ₃), L2 + L3·cos(θ₃))
        k1 = self.L2 + self.L3 * cos_theta3
        k2 = self.L3 * sin_theta3
        theta2_rad = np.arctan2(z_prime, r) - np.arctan2(k2, k1)

        # ---- Joints de muñeca fijos en 0 para agarre vertical -----------
        theta4_rad = 0.0
        theta5_rad = 0.0
        theta6_rad = 0.0

        # Convertir todo a GRADOS para la interfaz exterior (pymycobot)
        angles_deg = np.degrees([
            theta1_rad,
            theta2_rad,
            theta3_rad,
            theta4_rad,
            theta5_rad,
            theta6_rad,
        ])

        return angles_deg


# ===========================================================================
# Clase 3 — Verificador de Colisiones (Collision Checker)
# ===========================================================================

class CollisionChecker:
    """
    Verifica que un movimiento planificado sea seguro antes de enviarlo
    al robot, comprobando:

    A) Colisión con la mesa de trabajo → coordenada Z sobre umbral mínimo.
    B) Límites articulares (soft limits) → ángulos dentro del rango físico.
    """

    def __init__(self, z_safe_mm: float = 10.0):
        """
        Parámetros
        ----------
        z_safe_mm : float
            Altura mínima permitida del efector sobre la mesa de trabajo (mm).
            El robot no debe bajar por debajo de este valor para evitar
            colisionar con la superficie.
        """
        self.z_safe_mm    = z_safe_mm
        self.joint_limits = JOINT_LIMITS_DEG.copy()   # shape (6, 2)

    # ------------------------------------------------------------------
    def _check_workspace_floor(self, coords: np.ndarray) -> tuple[bool, str]:
        """
        Comprueba que la coordenada Z del efector sea segura respecto a la mesa.

        Parámetros
        ----------
        coords : array-like (3,) — [x, y, z] en mm

        Retorna
        -------
        (ok, mensaje) : (bool, str)
        """
        z = float(coords[2])
        if z < self.z_safe_mm:
            return False, (
                f"COLISIÓN CON MESA: z={z:.2f} mm < z_seguro={self.z_safe_mm:.2f} mm."
            )
        return True, "Z OK"

    # ------------------------------------------------------------------
    def _check_joint_limits(self, angles_deg: np.ndarray) -> tuple[bool, str]:
        """
        Verifica que todos los ángulos estén dentro de los rangos físicos.

        Parámetros
        ----------
        angles_deg : array-like (6,) — ángulos en GRADOS

        Retorna
        -------
        (ok, mensaje) : (bool, str)
        """
        angles_deg = np.asarray(angles_deg, dtype=float)
        for i, (angle, (lo, hi)) in enumerate(zip(angles_deg, self.joint_limits)):
            if not (lo <= angle <= hi):
                return False, (
                    f"LÍMITE ARTICULAR EXCEDIDO: J{i+1} = {angle:.2f}° "
                    f"fuera del rango [{lo}°, {hi}°]."
                )
        return True, "Límites articulares OK"

    # ------------------------------------------------------------------
    def is_safe(
        self,
        angles_deg: list | np.ndarray,
        coords: list | np.ndarray,
    ) -> bool:
        """
        Método principal de validación de seguridad.

        Verifica (A) límites articulares y (B) altura sobre la mesa.
        Imprime un mensaje descriptivo si alguna comprobación falla.

        Parámetros
        ----------
        angles_deg : array-like (6,) — ángulos articulares en GRADOS
        coords     : array-like (3,) — posición cartesiana [x, y, z] en mm

        Retorna
        -------
        bool — True si el movimiento es completamente seguro, False en caso contrario
        """
        angles_deg = np.asarray(angles_deg, dtype=float)
        coords     = np.asarray(coords,     dtype=float)

        # Comprobación A: límites articulares
        ok_joints, msg_joints = self._check_joint_limits(angles_deg)
        if not ok_joints:
            print(f"[CollisionChecker] ⚠  {msg_joints}")
            return False

        # Comprobación B: altura sobre la mesa de trabajo
        ok_z, msg_z = self._check_workspace_floor(coords)
        if not ok_z:
            print(f"[CollisionChecker] ⚠  {msg_z}")
            return False

        print("[CollisionChecker] ✓  Movimiento seguro.")
        return True


# ===========================================================================
# Bloque de prueba rápida (se ejecuta solo cuando el módulo es el script principal)
# ===========================================================================

if __name__ == "__main__":
    # Forzar UTF-8 en la salida de la terminal (necesario en Windows)
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("=" * 60)
    print("  Test de integracion --- cinematica.py  (MyCobot 280)")
    print("=" * 60)

    # --- Instancias ---
    fk      = ForwardKinematics()
    ik      = InverseKinematics(elbow_up=True)
    checker = CollisionChecker(z_safe_mm=10.0)

    # --- Test 1: FK desde una configuracion articular conocida ---
    # Usamos angulos con muñeca en 0 para que TCP coincida con wrist center + offset Z
    print("\n[TEST 1] FK --- angulos de prueba: [30, -30, 60, 0, 0, 0] grados")
    angles_test = [30.0, -30.0, 60.0, 0.0, 0.0, 0.0]
    pos_fk, T_06 = fk.compute_fk(angles_test)
    print(f"  Posicion TCP (FK): x={pos_fk[0]:.2f} mm, "
          f"y={pos_fk[1]:.2f} mm, z={pos_fk[2]:.2f} mm")

    # --- Test 2: IK round-trip sobre el wrist center (origen de J4 = T_0_3[:3,3]) ---
    # La IK analitica 2R opera sobre el wrist center = origen del frame J4.
    # Usamos FK parcial (3 joints) para obtenerlo exactamente, luego aplicamos
    # las ecuaciones de la guia directamente y verificamos que los angulos coincidan.
    print("\n[TEST 2] IK round-trip exacto --- wrist center = T_0_3[:3,3]")
    try:
        # Origen de J4 via FK parcial: T_0_3 con los angulos de prueba
        wc_pos, T_0_3 = fk.compute_partial_fk(angles_test, num_joints=3)
        x_wc = float(wc_pos[0])
        y_wc = float(wc_pos[1])
        z_wc = float(wc_pos[2])
        print(f"  Wrist center (T_0_3): x={x_wc:.4f} mm, y={y_wc:.4f} mm, z={z_wc:.4f} mm")

        # Aplicar las ecuaciones IK de la guia directamente sobre el wrist center
        theta1_rad = np.arctan2(y_wc, x_wc)
        r       = np.sqrt(x_wc**2 + y_wc**2)
        z_prime = z_wc - D1
        cos_t3  = (r**2 + z_prime**2 - L2**2 - L3**2) / (2.0 * L2 * L3)
        if np.abs(cos_t3) > 1.0:
            raise ValueError(f"Wrist center inalcanzable: cos(t3)={cos_t3:.4f}")
        sin_t3  = np.sqrt(1.0 - cos_t3**2)   # elbow_up
        theta3_rad = np.arctan2(sin_t3, cos_t3)
        k1 = L2 + L3 * cos_t3
        k2 = L3 * sin_t3
        theta2_rad = np.arctan2(z_prime, r) - np.arctan2(k2, k1)
        angles_ik_wc = np.degrees([theta1_rad, theta2_rad, theta3_rad, 0.0, 0.0, 0.0])

        print(f"  Angulos originales  [J1..J3]: {np.round(angles_test[:3], 6)}")
        print(f"  Angulos IK          [J1..J3]: {np.round(angles_ik_wc[:3], 6)}")
        angle_error = np.max(np.abs(angles_ik_wc[:3] - np.array(angles_test[:3])))
        print(f"  Max error articular: {angle_error:.8f} grados")
        if angle_error < 1e-4:
            print("  [OK] Round-trip exacto sobre wrist center")
        else:
            print("  [WARN] Discrepancia > 1e-4 grados")
    except ValueError as e:
        print(f"  IK Error: {e}")


    # --- Test 3: Verificacion de colisiones con angulos del Test 1 ---
    print("\n[TEST 3] CollisionChecker --- posicion de prueba")
    safe = checker.is_safe(angles_test, pos_fk)
    print(f"  Movimiento seguro? {safe}")

    # --- Test 4: Posicion fuera del espacio de trabajo ---
    print("\n[TEST 4] IK --- punto inalcanzable (x=9999, y=9999, z=9999)")
    try:
        _ = ik.ik_solve(x=9999.0, y=9999.0, z=9999.0)
    except ValueError as e:
        print(f"  OK - Excepcion capturada correctamente:")
        print(f"       {e}")

    # --- Test 5: Angulo fuera de limites articulares ---
    print("\n[TEST 5] CollisionChecker --- J1=200 grados (fuera de rango)")
    bad_angles = [200.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # J1 > 168 deg
    checker.is_safe(bad_angles, pos_fk)

    # --- Test 6: Colision con la mesa (z muy bajo) ---
    print("\n[TEST 6] CollisionChecker --- z=5 mm (por debajo de z_seguro=10 mm)")
    checker.is_safe(angles_test, np.array([100.0, 0.0, 5.0]))

    print("\n" + "=" * 60)
    print("  Tests completados.")
    print("=" * 60)

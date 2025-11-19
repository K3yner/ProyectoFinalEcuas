import numpy as np
from typing import Callable, Union, List, Tuple
import matplotlib.pyplot as plt

class RungeKutta4:
    """
    Clase para resolver ecuaciones diferenciales usando el método de Runge-Kutta de orden 4.
    """
    
    def __init__(self, f: Callable, t0: float, y0: Union[float, List[float]], 
                 h: float, n: int):
        self.f = f
        self.t0 = t0
        self.y0 = np.array(y0, dtype=float)
        self.h = h
        self.n = n
        
        # Verificar si es un sistema o una ecuación única
        self.is_system = isinstance(y0, (list, np.ndarray)) and len(self.y0) > 1
        
    def _rk4_step(self, t: float, y: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Realiza un paso del método de Runge-Kutta de orden 4.
        """
        k1 = self.h * self.f(t, y)
        k2 = self.h * self.f(t + self.h/2, y + k1/2)
        k3 = self.h * self.f(t + self.h/2, y + k2/2)
        k4 = self.h * self.f(t + self.h, y + k3)
        
        y_new = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        t_new = t + self.h
        
        return t_new, y_new
    
    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resuelve la ecuación diferencial usando Runge-Kutta de orden 4.
        """
        # Inicializar arrays para almacenar resultados
        tiempos = np.zeros(self.n + 1)
        soluciones = np.zeros((self.n + 1, len(self.y0)))
        
        # Condiciones iniciales
        tiempos[0] = self.t0
        soluciones[0] = self.y0
        
        # Iterar usando Runge-Kutta
        t_actual = self.t0
        y_actual = self.y0.copy()
        
        for i in range(1, self.n + 1):
            t_actual, y_actual = self._rk4_step(t_actual, y_actual)
            tiempos[i] = t_actual
            soluciones[i] = y_actual
            
        return tiempos, soluciones
    
    def plot_solution(self, tiempos: np.ndarray, soluciones: np.ndarray, 
                     labels: List[str] = None, title: str = "Solución de la Ecuación Diferencial"):
        """
        Grafica la solución de la ecuación diferencial.
        """
        plt.figure(figsize=(10, 6))
        
        if self.is_system:
            # Para sistemas, graficar cada variable por separado
            if labels is None:
                labels = [f'$y_{i}$' for i in range(soluciones.shape[1])]
            
            for i in range(soluciones.shape[1]):
                plt.plot(tiempos, soluciones[:, i], label=labels[i], linewidth=2)
        else:
            # Para una sola ecuación
            if labels is None:
                labels = ['y(t)']
            plt.plot(tiempos, soluciones, label=labels[0], linewidth=2)
        
        plt.xlabel('Tiempo (t)')
        plt.ylabel('Solución y(t)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

# CORRECCIÓN DEFINITIVA: Ecuación de primer orden
def CaidaFriccion(t: float, v: np.ndarray, k: float, m: float, g:float) -> np.ndarray:
    """ 
    Ecuación: dv/dt = -(k/m) * v +g 
    """
    # v es un array numpy, necesitamos extraer el valor escalar
    v_valor = v[0]  # Extraer el primer elemento del array
    return np.array([-(k/m) * v_valor+g])  # Devolver como array de 1 elemento

# CORRECCIÓN DEFINITIVA: Ecuación de segundo orden
def circuito_rlc_serie(t: float, y: np.ndarray, R: float, L: float, C: float) -> np.ndarray:
    """
    Circuito RLC serie sin fuente: L*d²q/dt² + R*dq/dt + (1/C)*q = 0
    
    Convertido a sistema:
        y[0] = q (carga)
        y[1] = dq/dt = i (corriente)
        
    Ecuaciones:
        dq/dt = y[1]
        d²q/dt² = (-R*y[1] - (1/C)*y[0]) / L
    """
    # Sistema de ecuaciones
    dq_dt = y[1]  # dq/dt = i (corriente)
    di_dt = (-R * y[1] - (1/C) * y[0]) / L  # d²q/dt²
    
    return np.array([dq_dt, di_dt])

# CORRECCIÓN DEFINITIVA: Sistema de ecuaciones
def ResortesAcoplados(t: float, y: np.ndarray, m1: float, m2: float, k1: float, k2: float) -> np.ndarray:
    """
    Sistema de resortes acoplados
    """
    # y es un array numpy [x1, v1, x2, v2]
    x1, v1, x2, v2 = y
    
    # Ecuaciones
    dx1_dt = v1
    dv1_dt = (-k1 * x1 + k2 * (x2 - x1)) / m1
    dx2_dt = v2
    dv2_dt = (-k2 * (x2 - x1)) / m2
    
    return np.array([dx1_dt, dv1_dt, dx2_dt, dv2_dt])

# EJEMPLOS CORREGIDOS

def ejemplo_caida_friccion():
    """Ejemplo 1: Caída con fricción (ecuación de primer orden)"""
    # Parámetros
    g= 9.81
    m = 1      # masa [kg]
    k = 0.5     # coeficiente de fricción
    t0 = 0.0     # tiempo inicial
    v0 = [0]  # velocidad inicial [m/s] - como lista
    h = 0.1      # paso de tiempo
    n = 50       # número de pasos
    
    # Definir la ecuación usando lambda para fijar parámetros
    def ecuacion_caida(t, v):
        return CaidaFriccion(t, v, k, m, g)
    
    # Resolver
    solver = RungeKutta4(ecuacion_caida, t0, v0, h, n)
    tiempos, soluciones = solver.solve()
    
    # Graficar
    solver.plot_solution(tiempos, soluciones, 
                        labels=['Velocidad [m/s]'],
                        title='Caída con Fricción: Velocidad vs Tiempo')
    
    return tiempos, soluciones

def ejemplo_circuito_rlc():
    """Ejemplo: Circuito RLC en serie (con carga q)"""
    # Parámetros del circuito
    R = 2.0      # Resistencia [Ω]
    L = 1.0      # Inductancia [H]
    C = 0.1      # Capacitancia [F]
    
    # Condiciones iniciales: [carga q, corriente i = dq/dt]
    condiciones_iniciales = [1.0, 0.0]  # q(0) = 1 C, i(0) = 0 A
    t0 = 0.0
    h = 0.01
    n = 500
    
    # Definir la ecuación
    def ecuacion_rlc(t, y):
        return circuito_rlc_serie(t, y, R, L, C)
    
    # Resolver
    solver = RungeKutta4(ecuacion_rlc, t0, condiciones_iniciales, h, n)
    tiempos, soluciones = solver.solve()
    
    # Graficar
    solver.plot_solution(tiempos, soluciones,
                        labels=['Carga q(t) [C]', 'Corriente i(t) [A]'],
                        title=f'Circuito RLC Serie (R={R}Ω, L={L}H, C={C}F)')
    
    return tiempos, soluciones

def ejemplo_resortes_acoplados():
    """Ejemplo 3: Resortes acoplados (sistema de ecuaciones)"""
    # Parámetros
    m1, m2 = 1.0, 1.    # masas [kg]
    k1, k2 = 10.0, 4.0    # constantes de resorte [N/m]
    
    # Condiciones iniciales: [x1, v1, x2, v2]
    condiciones_iniciales = [0.0, 1.0, 0.0, -1.0]
    t0 = 0.0
    h = 0.05
    n = 400
    
    # Definir el sistema
    def sistema_resortes(t, y):
        return ResortesAcoplados(t, y, m1, m2, k1, k2)
    
    # Resolver
    solver = RungeKutta4(sistema_resortes, t0, condiciones_iniciales, h, n)
    tiempos, soluciones = solver.solve()
    
    # Graficar
    plt.figure(figsize=(12, 8))
    
    # plt.subplot(2, 2, 1)
    plt.plot(tiempos, soluciones[:, 0], 'b-', label='Masa 1', linewidth=2)
    plt.plot(tiempos, soluciones[:, 2], 'r-', label='Masa 2', linewidth=2)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Posición [m]')
    plt.title('Posiciones vs Tiempo')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # plt.subplot(2, 2, 2)
    # plt.plot(soluciones[:, 0], soluciones[:, 1], 'b-', linewidth=1)
    # plt.xlabel('Posición masa 1 [m]')
    # plt.ylabel('Velocidad masa 1 [m/s]')
    # plt.title('Espacio de Fase - Masa 1')
    # plt.grid(True, alpha=0.3)
    
    # plt.subplot(2, 2, 3)
    # plt.plot(soluciones[:, 2], soluciones[:, 3], 'r-', linewidth=1)
    # plt.xlabel('Posición masa 2 [m]')
    # plt.ylabel('Velocidad masa 2 [m/s]')
    # plt.title('Espacio de Fase - Masa 2')
    # plt.grid(True, alpha=0.3)
    
    # plt.subplot(2, 2, 4)
    # plt.plot(soluciones[:, 0], soluciones[:, 2], 'g-', linewidth=1)
    # plt.xlabel('Posición masa 1 [m]')
    # plt.ylabel('Posición masa 2 [m]')
    # plt.title('Trayectoria en el Espacio de Configuración')
    # plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return tiempos, soluciones

# Ejecutar todos los ejemplos
if __name__ == "__main__":
    print("Ejemplo 1: Caída con fricción")
    tiempos_caida, velocidades = ejemplo_caida_friccion()
    
    print("\nEjemplo 2: Circuito RLC")
    tiempos_rlc, estados_rlc = ejemplo_circuito_rlc()
    
    print("\nEjemplo 3: Resortes acoplados")
    tiempos_resortes, estados_resortes = ejemplo_resortes_acoplados()
    
    print("¡Todos los ejemplos ejecutados correctamente!")
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
                     labels: List[str] = None, 
                     title: str = "Solución de la Ecuación Diferencial",
                     analitica: Callable = None):
        """
        Grafica la solución de la ecuación diferencial.
        
        Args:
            tiempos: Array de tiempos
            soluciones: Array de soluciones
            labels: Lista de etiquetas para cada variable
            title: Título del gráfico
            analitica: Función que devuelve la solución analítica (opcional)
        """
        if self.is_system and soluciones.shape[1] >= 2:
            # Para sistemas con 2 o más variables, crear subgráficas
            n_vars = soluciones.shape[1]
            fig, axes = plt.subplots(n_vars, 1, figsize=(10, 4*n_vars))
            
            if n_vars == 1:
                axes = [axes]
            
            if labels is None:
                labels = [f'$y_{i}$' for i in range(n_vars)]
            
            for i in range(n_vars):
                # Graficar solución numérica
                axes[i].plot(tiempos, soluciones[:, i], 'b-', 
                           label='Aproximación por Runge-Kutta', linewidth=2)
                
                # Graficar solución analítica si está disponible
                if analitica is not None:
                    y_analitica = analitica(tiempos)
                    if isinstance(y_analitica, (list, np.ndarray)) and len(y_analitica.shape) > 1:
                        # Si la solución analítica es un array multidimensional
                        axes[i].plot(tiempos, y_analitica[:, i], 'r--', 
                                   label='Solución analítica', linewidth=2, alpha=0.8)
                    else:
                        # Si es un array unidimensional
                        axes[i].plot(tiempos, y_analitica, 'r--', 
                                   label='Solución analítica', linewidth=2, alpha=0.8)
                
                axes[i].set_xlabel('Tiempo (t)')
                axes[i].set_ylabel(labels[i])
                axes[i].set_title(f'{title} - {labels[i]}')
                axes[i].grid(True, alpha=0.3)
                axes[i].legend()
            
            plt.tight_layout()
            plt.show()
            
        else:
            # Para una sola ecuación o sistemas pequeños en una sola gráfica
            plt.figure(figsize=(10, 6))
            
            if self.is_system:
                # Para sistemas, graficar cada variable por separado
                if labels is None:
                    labels = [f'$y_{i}$' for i in range(soluciones.shape[1])]
                
                for i in range(soluciones.shape[1]):
                    # Graficar solución numérica
                    plt.plot(tiempos, soluciones[:, i], 
                           label=f'{labels[i]} - Aproximación por Runge-Kutta', 
                           linewidth=2)
                    
                    # Graficar solución analítica si está disponible
                    if analitica is not None:
                        y_analitica = analitica(tiempos)
                        if isinstance(y_analitica, (list, np.ndarray)) and len(y_analitica.shape) > 1:
                            plt.plot(tiempos, y_analitica[:, i], '--', 
                                   label=f'{labels[i]} - Solución analítica', 
                                   linewidth=2, alpha=0.8)
                        else:
                            plt.plot(tiempos, y_analitica, '--', 
                                   label=f'{labels[i]} - Solución analítica', 
                                   linewidth=2, alpha=0.8)
            else:
                # Para una sola ecuación
                if labels is None:
                    labels = ['y(t)']
                
                # Graficar solución numérica
                plt.plot(tiempos, soluciones, 'b-', 
                       label='Aproximación por Runge-Kutta', linewidth=2)
                
                # Graficar solución analítica si está disponible
                if analitica is not None:
                    y_analitica = analitica(tiempos)
                    plt.plot(tiempos, y_analitica, 'r--', 
                           label='Solución analítica', linewidth=2, alpha=0.8)
            
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
    
    # Solución analítica para caída con fricción
    def solucion_analitica_caida(t):
        v_terminal = g * m / k  # velocidad terminal
        return v_terminal * (1 - np.exp(-k/m * t))
    
    # Resolver
    solver = RungeKutta4(ecuacion_caida, t0, v0, h, n)
    tiempos, soluciones = solver.solve()
    
    # Graficar
    solver.plot_solution(tiempos, soluciones, 
                        labels=['Velocidad [m/s]'],
                        title='Caída con Fricción: Velocidad vs Tiempo',
                        analitica=solucion_analitica_caida)
    
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

    def solAnalitica(t):
        return np.exp(-t)*(np.cos(3*t)+np.sin(3*t))
    
    # Graficar
    solver.plot_solution(tiempos, soluciones,
                        labels=['Carga q(t) [C]', 'Corriente i(t) [A]'],
                        title=f'Circuito RLC Serie (R={R}Ω, L={L}H, C={C}F)',
                        analitica = solAnalitica)
    
    return tiempos, soluciones

def ejemplo_resortes_acoplados():
    """Ejemplo 3: Resortes acoplados (sistema de ecuaciones)"""
    # Parámetros
    m1, m2 = 1.0, 1.0    # masas [kg]
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
    
    # Soluciones analíticas
    y1_ex = (-np.sqrt(2)/10)*np.sin(np.sqrt(2)*tiempos) + (np.sqrt(3)/5)*np.sin(2*np.sqrt(3)*tiempos)
    y2_ex = (-np.sqrt(2)/5)*np.sin(np.sqrt(2)*tiempos) - (np.sqrt(3)/10)*np.sin(2*np.sqrt(3)*tiempos)

    # Graficar en 2 subgráficas
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Gráfica para Masa 1
    ax1.plot(tiempos, soluciones[:, 0], 'b-', label='Aproximación por Runge-Kutta', linewidth=2)
    ax1.plot(tiempos, y1_ex, 'r--', label='Solución analítica', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Tiempo [s]')
    ax1.set_ylabel('Posición [m]')
    ax1.set_title('Masa 1 - Posición vs Tiempo')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfica para Masa 2
    ax2.plot(tiempos, soluciones[:, 2], 'b-', label='Aproximación por Runge-Kutta', linewidth=2)
    ax2.plot(tiempos, y2_ex, 'r--', label='Solución analítica', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Tiempo [s]')
    ax2.set_ylabel('Posición [m]')
    ax2.set_title('Masa 2 - Posición vs Tiempo')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
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
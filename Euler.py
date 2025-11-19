import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Union, Tuple
import warnings

class EulerSolver:
    """
    Clase para resolver ecuaciones diferenciales usando el método de Euler.
    
    Esta clase puede resolver:
    - Ecuaciones diferenciales de primer orden
    - Ecuaciones diferenciales de segundo orden
    - Sistemas de ecuaciones diferenciales
    """
    
    def __init__(self):
        """Inicializa el solucionador de Euler."""
        self.solution = None
        self.t_values = None
        
    def solve_first_order(self, 
                         f: Callable[[float, float], float], 
                         y0: float, 
                         t_span: Tuple[float, float], 
                         n_steps: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resuelve una ecuación diferencial de primer orden.
        
        Parameters:
        -----------
        f : Callable
            Función que define la EDO: dy/dt = f(t, y)
        y0 : float
            Condición inicial y(t0)
        t_span : Tuple[float, float]
            Intervalo de tiempo (t0, tf)
        n_steps : int
            Número de pasos de tiempo
            
        Returns:
        --------
        t_values : np.ndarray
            Array de valores de tiempo
        y_values : np.ndarray
            Array de soluciones
        """
        t0, tf = t_span
        h = (tf - t0) / n_steps
        
        # Inicializar arrays
        t_values = np.linspace(t0, tf, n_steps + 1)
        y_values = np.zeros(n_steps + 1)
        y_values[0] = y0
        
        # Método de Euler
        for i in range(n_steps):
            y_values[i + 1] = y_values[i] + h * f(t_values[i], y_values[i])
        
        self.solution = y_values
        self.t_values = t_values
        
        return t_values, y_values
    
    def solve_second_order(self, 
                          f: Callable[[float, float, float], float], 
                          y0: float, 
                          v0: float, 
                          t_span: Tuple[float, float], 
                          n_steps: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Resuelve una ecuación diferencial de segundo orden.
        
        Parameters:
        -----------
        f : Callable
            Función que define la EDO: d²y/dt² = f(t, y, dy/dt)
        y0 : float
            Condición inicial para y(t0)
        v0 : float
            Condición inicial para dy/dt(t0)
        t_span : Tuple[float, float]
            Intervalo de tiempo (t0, tf)
        n_steps : int
            Número de pasos de tiempo
            
        Returns:
        --------
        t_values : np.ndarray
            Array de valores de tiempo
        y_values : np.ndarray
            Array de posiciones
        v_values : np.ndarray
            Array de velocidades
        """
        t0, tf = t_span
        h = (tf - t0) / n_steps
        
        # Inicializar arrays
        t_values = np.linspace(t0, tf, n_steps + 1)
        y_values = np.zeros(n_steps + 1)
        v_values = np.zeros(n_steps + 1)
        
        y_values[0] = y0
        v_values[0] = v0
        
        # Método de Euler para sistema de primer orden
        for i in range(n_steps):
            t = t_values[i]
            y = y_values[i]
            v = v_values[i]
            
            # Actualizar posición y velocidad
            y_values[i + 1] = y + h * v
            v_values[i + 1] = v + h * f(t, y, v)
        
        self.solution = (y_values, v_values)
        self.t_values = t_values
        
        return t_values, y_values, v_values
    
    def solve_system(self, 
                    f_system: Callable[[float, np.ndarray], np.ndarray], 
                    y0: np.ndarray, 
                    t_span: Tuple[float, float], 
                    n_steps: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resuelve un sistema de ecuaciones diferenciales.
        
        Parameters:
        -----------
        f_system : Callable
            Función que define el sistema: dy/dt = f_system(t, y)
            donde y es un vector
        y0 : np.ndarray
            Condiciones iniciales
        t_span : Tuple[float, float]
            Intervalo de tiempo (t0, tf)
        n_steps : int
            Número de pasos de tiempo
            
        Returns:
        --------
        t_values : np.ndarray
            Array de valores de tiempo
        y_values : np.ndarray
            Array de soluciones (cada fila es una variable)
        """
        t0, tf = t_span
        h = (tf - t0) / n_steps
        n_vars = len(y0)
        
        # Inicializar arrays
        t_values = np.linspace(t0, tf, n_steps + 1)
        y_values = np.zeros((n_steps + 1, n_vars))
        y_values[0, :] = y0
        
        # Método de Euler para sistemas
        for i in range(n_steps):
            t = t_values[i]
            y_current = y_values[i, :]
            
            y_values[i + 1, :] = y_current + h * f_system(t, y_current)
        
        self.solution = y_values
        self.t_values = t_values
        
        return t_values, y_values
    
    def plot_solution(self, 
                     labels: List[str] = None, 
                     title: str = "Solución de la Ecuación Diferencial",
                     figsize: Tuple[float, float] = (10, 6)) -> None:
        """
        Grafica la solución obtenida.
        
        Parameters:
        -----------
        labels : List[str], optional
            Etiquetas para las curvas
        title : str
            Título del gráfico
        figsize : Tuple[float, float]
            Tamaño de la figura
        """
        if self.solution is None or self.t_values is None:
            raise ValueError("No hay solución para graficar. Ejecuta primero un método solve.")
        
        plt.figure(figsize=figsize)
        
        if isinstance(self.solution, tuple):
            # Caso de segundo orden
            y_values, v_values = self.solution
            plt.plot(self.t_values, y_values, label='Posición (y)')
            plt.plot(self.t_values, v_values, label='Velocidad (v)')
        elif self.solution.ndim == 1:
            # Caso de primer orden
            if labels is None:
                labels = ['y(t)']
            plt.plot(self.t_values, self.solution, label=labels[0])
        else:
            # Caso de sistema
            n_vars = self.solution.shape[1]
            if labels is None:
                labels = [f'y{i+1}(t)' for i in range(n_vars)]
            
            for i in range(n_vars):
                plt.plot(self.t_values, self.solution[:, i], label=labels[i])
        
        plt.xlabel('Tiempo')
        plt.ylabel('Solución')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


#ED de primer orden ---------------------------------------------------------------------------------------

def caida_libre_con_friccion(masa: float = 1.0, 
                            gravedad: float = 9.81, 
                            coeficiente_friccion: float = 0.1) -> Callable[[float, np.ndarray], np.ndarray]:
    """
    Define el sistema de ecuaciones para caída libre con fricción del aire.
    
    Sistema basado en:
    dv/dt = (k/m) * v
    
    Considerando que la fricción se opone al movimiento:
    dy/dt = v
    dv/dt = g - (k/m) * v
    
    Parameters:
    -----------
    masa : float
        Masa del objeto (kg)
    gravedad : float
        Aceleración gravitacional (m/s²)
    coeficiente_friccion : float
        Coeficiente de fricción (kg/s)
        
    Returns:
    --------
    Callable: Función del sistema para usar con solve_system
        
    Uso de condiciones iniciales:
    ----------------------------
    Las condiciones iniciales deben ser un array numpy con 2 elementos en este orden:
    y0 = [y_0, v_0]
    
    donde:
    - y_0: altura inicial del objeto (m)
    - v_0: velocidad inicial del objeto (m/s)
    """
    def sistema(t, y):
        pos, vel = y
        dpos_dt = vel
        dvel_dt = gravedad - (coeficiente_friccion / masa) * vel
        return np.array([dpos_dt, dvel_dt])
    
    return sistema


#ED de segundo orden ---------------------------------------------------------------------------------------

def circuito_rlc_serie(R: float , 
                      L: float , 
                      C: float,
                      V0: Callable = None) -> Callable[[float, np.ndarray], np.ndarray]:
    """
    Define el sistema de ecuaciones para un circuito RLC en serie.
    
    Sistema para carga del capacitor:
    dq/dt = i
    di/dt = (V(t) - R*i - q/C) / L
    
    Parameters:
    -----------
    R : float
        R (Ohms)
    L : float
        L (Henries)
    C : float
        C (Farads)
    V0 : Callable
        Función V(t) que describe el voltaje de la fuente
        
    Returns:
    --------
    Callable: Función del sistema para usar con solve_system
    """
    if V0 is None:
        # Por defecto, fuente de DC
        V0 = lambda t: 1.0 if t >= 0 else 0.0
    
    def sistema(t, y):
        q, i = y  # q: carga del capacitor, i: corriente
        dq_dt = i
        di_dt = (V0(t) - R * i - q / C) / L
        return np.array([dq_dt, di_dt])
    
    return sistema

#Sistema 2x2 de ED de primer orden ---------------------------------------------------------------------------------------

def resortes_acoplados(m1: float = 1.0, 
                      m2: float = 1.0, 
                      k1: float = 10.0, 
                      k2: float = 10.0) -> Callable[[float, np.ndarray], np.ndarray]:
    """
    Define el sistema de ecuaciones para dos resortes acoplados.
    
    Sistema basado en:
    m₁x₁'' = -k₁x₁ + k₂(x₂ - x₁)
    m₂x₂'' = -k₂(x₂ - x₁)
    
    Convertido a sistema de primer orden:
    dx₁/dt = v₁
    dv₁/dt = (-k₁x₁ + k₂(x₂ - x₁)) / m₁
    dx₂/dt = v₂
    dv₂/dt = (-k₂(x₂ - x₁)) / m₂
    
    Parameters:
    -----------
    m1, m2 : float
        Masas de los dos objetos
    k1, k2 : float
        Constantes de los resortes

    Uso de condiciones iniciales:
    ----------------------------
    Las condiciones iniciales deben ser un array numpy con 4 elementos en este orden:
    y0 = [x1_0, v1_0, x2_0, v2_0]
        
    Returns:
    --------
    Callable: Función del sistema para usar con solve_system
    """
    def sistema(t, y):
        x1, v1, x2, v2 = y
        dx1_dt = v1
        dv1_dt = (-k1 * x1 + k2 * (x2 - x1)) / m1
        dx2_dt = v2
        dv2_dt = (-k2 * (x2 - x1)) / m2
        return np.array([dx1_dt, dv1_dt, dx2_dt, dv2_dt])
    
    return sistema





#Ejemplos con valores concretos---------------------------------------------------
def ejemplo_caida_libre():
    """
    Ejemplo: Caída libre con fricción del aire usando dv/dt = g - (k/m)v
    """
    print("\n=== Ejemplo: Caída libre con fricción ===")
    
    # Parámetros del sistema
    masa = 1.0           # kg
    gravedad = 9.81      # m/s²
    coeficiente_friccion = 0.5  # kg/s
    
    # Condiciones iniciales: [posición, velocidad]
    y0 = np.array([100.0, 0.0])  # Desde 100m de altura, en reposo
    
    # Crear el sistema
    sistema = caida_libre_con_friccion(masa, gravedad, coeficiente_friccion)
    
    # Resolver
    solver = EulerSolver()
    t, sol = solver.solve_system(sistema, y0, t_span=(0, 25), n_steps=2000)
    
    # Extraer posición y velocidad
    posicion = sol[:, 0]
    velocidad = sol[:, 1]

    #solución analítica
    y_ex= 19.6*(1-np.exp(-0.5*t))
    
    # Graficar
    plt.figure(figsize=(12, 8))
    
    # plt.subplot(2, 2, 1)
    # plt.plot(t, posicion, 'b-', linewidth=2)
    # plt.xlabel('Tiempo (s)')
    # plt.ylabel('Altura (m)')
    # plt.title('Altura vs Tiempo')
    # plt.grid(True, alpha=0.3)
    
    # plt.subplot(2, 2, 2)
    plt.plot(t, velocidad, 'r-',label="Aproximación por Euler", linewidth=2)
    plt.plot(t,y_ex, "b--", label = "analítica", linewidth=2)
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Velocidad (m/s)')
    plt.title('Velocidad vs Tiempo')
    plt.grid(True, alpha=0.3)
    
    # plt.subplot(2, 2, 3)
    # plt.plot(t, velocidad, 'g-', linewidth=2)
    # plt.xlabel('Tiempo (s)')
    # plt.ylabel('Velocidad (m/s)')
    # plt.title('Velocidad Terminal')
    # plt.grid(True, alpha=0.3)
    
    # Calcular velocidad terminal teórica
    v_terminal = (masa * gravedad) / coeficiente_friccion
    plt.axhline(y=v_terminal, color='k', linestyle='--', 
                label=f'Vel. terminal = {v_terminal:.2f} m/s')
    plt.legend()
    
    # plt.subplot(2, 2, 4)
    # plt.plot(posicion, velocidad, 'purple', linewidth=2)
    # plt.xlabel('Altura (m)')
    # plt.ylabel('Velocidad (m/s)')
    # plt.title('Diagrama de Fase: Velocidad vs Altura')
    # plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Velocidad terminal teórica: {v_terminal:.2f} m/s")
    print(f"Velocidad final numérica: {velocidad[-1]:.2f} m/s")
    print(f"Tiempo hasta llegar al suelo: {t[np.where(posicion <= 0)[0][0]]:.2f} s" if any(posicion <= 0) else "Objeto no ha llegado al suelo")

def ejemplo_circuito_rlc():
    """
    Ejemplo: Circuito RLC en serie subamortiguado.
    """
    print("\n=== Ejemplo: Circuito RLC en serie ===")
    
    # Parámetros del circuito (valores para oscilación subamortiguada)
    R = 2.0      # Ohms (baja resistencia para oscilaciones)
    L = 1.0      # Henries
    C = 0.1     # Farads
    
    # Condiciones iniciales: [carga del capacitor, corriente]
    y0 = np.array([1.0, 0.0])  # Capacitor descargado, sin corriente inicial
    
    # Crear sistema con fuente de DC de 1V que se enciende en t=0
    
    
    sistema = circuito_rlc_serie(R, L, C, None)
    
    # Resolver
    solver = EulerSolver()
    t, sol = solver.solve_system(sistema, y0, t_span=(0, 5), n_steps=4000)
    
    # Extraer carga y corriente
    carga = sol[:, 0]
    corriente = sol[:, 1]
    voltaje_capacitor = carga / C

    #Solución analítica
    y_ex= np.exp(-t)*(np.cos(3*t)+np.sin(3*t))
    
    # Graficar
    plt.figure(figsize=(12, 8))
    
    # plt.subplot(2, 2, 1)
    # plt.plot(t, voltaje_capacitor, 'b-', linewidth=2)
    # plt.xlabel('Tiempo (s)')
    # plt.ylabel('Voltaje en capacitor (V)')
    # plt.title('Voltaje en el Capacitor')
    # plt.grid(True, alpha=0.3)
    
    # plt.subplot(2, 1, 1)
    # plt.plot(t, corriente, 'r-', linewidth=2)
    # plt.plot(t,y_ex, "b--", label = "analítica", linewidth=2)
    # plt.xlabel('Tiempo (s)')
    # plt.ylabel('Corriente (A)')
    # plt.title('Corriente en el Circuito')
    # plt.grid(True, alpha=0.3)
    
    # plt.subplot(2, 1, 2)
    plt.plot(t, carga, 'g-',label="aproximación por Euler", linewidth=2)
    plt.plot(t,y_ex, "b--", label = "analítica", linewidth=2)
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Carga (C)')
    plt.title('Carga del Capacitor')
    plt.grid(True, alpha=0.3)
    
    # plt.subplot(2, 2, 4)
    # plt.plot(voltaje_capacitor, corriente, 'purple', linewidth=1.5)
    # plt.xlabel('Voltaje en capacitor (V)')
    # plt.ylabel('Corriente (A)')
    # plt.title('Diagrama de Fase: I vs Vc')
    # plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def ejemplo_resortes_acoplados():
    """
    Ejemplo: Sistema de dos resortes acoplados.
    """
    print("\n=== Ejemplo: Resortes Acoplados ===")
    
    # Parámetros del sistema
    m1, m2 = 1.0, 1.0           # Masas iguales
    k1, k2 = 10.0, 4.0         # Resortes iguales
    
    # Condiciones iniciales: [x1, v1, x2, v2]
    # Masa 1 desplazada, masa 2 en reposo
    y0 = np.array([0.0, 1.0, 0.0, -1.0])
    
    # Crear sistema
    sistema = resortes_acoplados(m1, m2, k1, k2)
    
    # Resolver
    solver = EulerSolver()
    t, sol = solver.solve_system(sistema, y0, t_span=(0, 20), n_steps=4000)
    
    # Extraer posiciones y velocidades
    x1 = sol[:, 0]
    v1 = sol[:, 1]
    x2 = sol[:, 2]
    v2 = sol[:, 3]

    #Solución analítica
    y1_ex= (-np.sqrt(2)/10)*np.sin(np.sqrt(2)*t)+(np.sqrt(3)/5)*np.sin(2*np.sqrt(3)*t)
    y2_ex= (-np.sqrt(2)/5)*np.sin(np.sqrt(2)*t)-(np.sqrt(3)/10)*np.sin(2*np.sqrt(3)*t)
    
    # Graficar
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, x1, 'b-', linewidth=2, label='Masa 1')
    plt.plot(t,y1_ex, label= "analítica", linewidth=2)
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Posición (m)')
    plt.title('Posición masa 1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(t,y2_ex, label= "analítica", linewidth=2)
    plt.plot(t, x2, 'r-', linewidth=2, label='aproximación por euler')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Posición (m)')
    plt.title('Posición masa 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # plt.subplot(3, 1, 2)
    # plt.plot(t, v1, 'b--', linewidth=2, label='Velocidad 1')
    # plt.plot(t, v2, 'r--', linewidth=2, label='Velocidad 2')
    # plt.xlabel('Tiempo (s)')
    # plt.ylabel('Velocidad (m/s)')
    # plt.title('Velocidades de las Masas')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    
    # plt.subplot(3, 1, 3)
    # plt.plot(x1, x2, 'purple', linewidth=1.5)
    # plt.xlabel('Posición Masa 1 (m)')
    # plt.ylabel('Posición Masa 2 (m)')
    # plt.title('Diagrama de Fase: x1 vs x2')
    # plt.grid(True, alpha=0.3)
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Calcular modos normales teóricos
    # Para el sistema: m₁ = m₂ = m, k₁ = k₂ = k
    # Modo 1: ω₁ = √(k/m) - masas se mueven en fase
    # Modo 2: ω₂ = √(3k/m) - masas se mueven en oposición
    omega1 = np.sqrt(k1 / m1)
    omega2 = np.sqrt(3 * k1 / m1)  # Para k₁ = k₂
    print(f"Frecuencia modo normal 1 (en fase): {omega1:.2f} rad/s")
    print(f"Frecuencia modo normal 2 (oposición): {omega2:.2f} rad/s")


def main():
    """Función principal que ejecuta todos los ejemplos."""
    print("Solver de Ecuaciones Diferenciales usando Método de Euler")
    print("=" * 60)
    
    # Ejecutar ejemplos de los nuevos sistemas
    ejemplo_caida_libre()
    ejemplo_circuito_rlc()
    ejemplo_resortes_acoplados()

if __name__ == "__main__":
    main()
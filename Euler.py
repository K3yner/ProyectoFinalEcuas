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


# Funciones de ejemplo para demostrar el uso
def ejemplo_primer_orden():
    """
    Ejemplo: Resuelve dy/dt = -2y, y(0) = 1
    Solución exacta: y(t) = e^(-2t)
    """
    print("=== Ejemplo: Ecuación de primer orden ===")
    
    def f(t, y):
        return -2 * y
    
    solver = EulerSolver()
    t, y = solver.solve_first_order(f, y0=1, t_span=(0, 5), n_steps=1000)
    
    # Solución exacta para comparación
    y_exacta = np.exp(-2 * t)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, y, label='Euler (numérica)', linewidth=2)
    plt.plot(t, y_exacta, 'r--', label='Exacta', linewidth=1.5)
    plt.xlabel('Tiempo')
    plt.ylabel('y(t)')
    plt.title('Solución: dy/dt = -2y, y(0) = 1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    error = np.abs(y - y_exacta).max()
    print(f"Error máximo: {error:.6f}")


def ejemplo_segundo_orden():
    """
    Ejemplo: Resuelve d²y/dt² + 4y = 0, y(0)=1, y'(0)=0
    Solución exacta: y(t) = cos(2t)
    """
    print("\n=== Ejemplo: Ecuación de segundo orden ===")
    
    def f(t, y, v):
        return -4 * y  # d²y/dt² = -4y
    
    solver = EulerSolver()
    t, y, v = solver.solve_second_order(f, y0=1, v0=0, t_span=(0, 10), n_steps=2000)
    
    # Solución exacta para comparación
    y_exacta = np.cos(2 * t)
    v_exacta = -2 * np.sin(2 * t)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, y, label='Euler (posición)', linewidth=2)
    plt.plot(t, y_exacta, 'r--', label='Exacta (posición)', linewidth=1.5)
    plt.ylabel('Posición')
    plt.title('Oscilador armónico: d²y/dt² + 4y = 0')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(t, v, label='Euler (velocidad)', linewidth=2)
    plt.plot(t, v_exacta, 'r--', label='Exacta (velocidad)', linewidth=1.5)
    plt.xlabel('Tiempo')
    plt.ylabel('Velocidad')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def ejemplo_sistema():
    """
    Ejemplo: Sistema de Lotka-Volterra (depredador-presa)
    dx/dt = αx - βxy
    dy/dt = δxy - γy
    """
    print("\n=== Ejemplo: Sistema de ecuaciones ===")
    
    def lotka_volterra(t, y):
        x, y_pred = y
        alpha, beta, delta, gamma = 1.0, 0.1, 0.075, 1.5
        dxdt = alpha * x - beta * x * y_pred
        dydt = delta * x * y_pred - gamma * y_pred
        return np.array([dxdt, dydt])
    
    solver = EulerSolver()
    t, sol = solver.solve_system(lotka_volterra, 
                                y0=np.array([20, 5]), 
                                t_span=(0, 50), 
                                n_steps=5000)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, sol[:, 0], label='Presas (x)', linewidth=2)
    plt.plot(t, sol[:, 1], label='Depredadores (y)', linewidth=2)
    plt.ylabel('Población')
    plt.title('Modelo Lotka-Volterra')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(sol[:, 0], sol[:, 1], 'b-', linewidth=1.5)
    plt.xlabel('Presas')
    plt.ylabel('Depredadores')
    plt.title('Diagrama de fase')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """Función principal que ejecuta todos los ejemplos."""
    print("Solver de Ecuaciones Diferenciales usando Método de Euler")
    print("=" * 60)
    
    # Ejecutar ejemplos
    ejemplo_primer_orden()
    ejemplo_segundo_orden()
    ejemplo_sistema()


if __name__ == "__main__":
    main()
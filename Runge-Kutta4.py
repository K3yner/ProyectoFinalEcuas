import numpy as np
from typing import Callable, Union, List, Tuple
import matplotlib.pyplot as plt

class RungeKutta4:
    """
    Clase para resolver ecuaciones diferenciales usando el método de Runge-Kutta de orden 4.
    
    Esta clase puede resolver tanto ecuaciones diferenciales individuales como sistemas
    de ecuaciones diferenciales.
    
    Atributos:
        f (Callable): Función que define la ecuación diferencial o sistema
        t0 (float): Valor inicial del tiempo
        y0 (Union[float, List[float]]): Condición inicial o condiciones iniciales
        h (float): Tamaño del paso
        n (int): Número de pasos
    """
    
    def __init__(self, f: Callable, t0: float, y0: Union[float, List[float]], 
                 h: float, n: int):
        """
        Inicializa el solucionador de Runge-Kutta.
        
        Args:
            f: Función que define la ecuación diferencial. Para sistemas, debe tomar
               (t, y) donde y es un array y retornar un array de la misma dimensión.
            t0: Valor inicial del tiempo
            y0: Condición inicial. Puede ser un número para una ecuación o una lista
                para sistemas.
            h: Tamaño del paso
            n: Número de pasos a calcular
        """
        self.f = f
        self.t0 = t0
        self.y0 = np.array(y0, dtype=float)
        self.h = h
        self.n = n
        
        # Verificar si es un sistema o una ecuación única
        self.is_system = isinstance(y0, (list, np.ndarray)) and len(y0) > 1
        
    def _rk4_step(self, t: float, y: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Realiza un paso del método de Runge-Kutta de orden 4.
        
        Args:
            t: Tiempo actual
            y: Valor actual de la solución
            
        Returns:
            Tupla (nuevo_tiempo, nueva_solucion)
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
        
        Returns:
            Tupla (tiempos, soluciones) donde:
                tiempos: Array con los valores de tiempo
                soluciones: Array con las soluciones en cada tiempo
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
        
        Args:
            tiempos: Array de tiempos
            soluciones: Array de soluciones
            labels: Lista de etiquetas para las curvas
            title: Título del gráfico
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



#Ecuación de primer orden ------------------------------------------------------------------------------------------------

def CaidaFriccion(t:float, v:np.ndarray, k:float, m:float) -> np.ndarray:

    """ 
    Ecuación:
        dv/dt = (k/m) * v

        m = masa del cuerpo
        k = 
        v = velocidad
    """

    return np.array([(k/m)*v])

#Ecuación de segundo orden --------------------------------------------------------------------------------------------
def circuito_rlc_serie(t: float, y: np.ndarray, R: float, L: float, C:float) -> np.ndarray:
    """
    Circuito RLC serie sin fuente: L*d²i/dt² + R*di/dt + (1/C)*i = 0
    
    Convertido a sistema:
        y[0] = i (corriente)
        y[1] = di/dt (derivada de la corriente)
        
    Ecuaciones:
        di/dt = y[1]
        d²i/dt² = (-R*y[1] - (1/C)*y[0]) / L
    """
    # Parámetros del circuito - puedes modificar estos valores
    #R Resistencia [Ω] - AMORTIGUAMIENTO CRÍTICO: R = 2√(L/C) ≈ 14.14Ω
    #L Inductancia [H]
    #C Capacitancia [F]
    
    # Sistema de ecuaciones (sin término de fuente)
    di_dt = y[1]  # di/dt = v
    dv_dt = (-R * y[1] - (1/C) * y[0]) / L  # d²i/dt²
    
    return np.array([di_dt, dv_dt])


#Sistema 2x2 de ecuaciones ------------------------------------------------------------------------------------------------------

def ResortesAcoplados(t: float, y: np.ndarray, m1:float, m2:float, k1:float, k2:float)-> np.ndarray:
    """
    Sistema 2x2 de ecuaciones diferenciales acopladas
    y = [x1, v1, x2, v2]
    
    Ecuaciones:
        dx1/dt = v1
        dv1/dt = (-k1*x1 + k2*(x2 - x1)) / m1
        dx2/dt = v2  
        dv2/dt = (-k2*(x2 - x1)) / m2
    """
    # Parámetros
    #m1, m2 = masas
    #k1, k2 = constantes de resorte
    
    # Extraer variables
    x1, v1, x2, v2 = y
    
    # Definir las 4 ecuaciones
    dx1_dt = v1  # dx1/dt = v1
    dv1_dt = (-k1*x1 + k2*(x2 - x1)) / m1  # dv1/dt = aceleración masa 1
    dx2_dt = v2  # dx2/dt = v2
    dv2_dt = (-k2*(x2 - x1)) / m2  # dv2/dt = aceleración masa 2
    
    return np.array([dx1_dt, dv1_dt, dx2_dt, dv2_dt])
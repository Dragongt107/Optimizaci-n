import numpy as np
import matplotlib.pyplot as plt

def metodo_secante(f, df, x1, x2, epsilon=0.01):
    """
    Implementa el Método de la Secante para encontrar un punto donde df(x) = 0,
    que corresponde a un punto crítico de la función f.

    Args:
        f (function): La función original.
        df (function): La primera derivada de la función.
        x1 (float): El primer punto inicial.
        x2 (float): El segundo punto inicial.
        epsilon (float): La tolerancia para la condición de terminación.

    Returns:
        tuple: Una tupla que contiene:
               - El punto crítico encontrado (donde df(x) ≈ 0).
               - El valor de la función en ese punto.
               - Una figura de Matplotlib con la gráfica del proceso.
               Devuelve (None, None, None) si el método no converge.
    """
    # Step 1: Inicialización
    
    # Preparar la figura para la visualización del proceso
    x_range = np.linspace(min(x1, x2) - 1, max(x1, x2) + 1, 500)
    y_values = f(x_range)
    
    fig, ax = plt.subplots()
    ax.plot(x_range, y_values, label='f(x)', color='blue')
    ax.set_title('Método de la Secante')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.grid(True)
    ax.legend()

    # Bucle principal del método
    while True:
        f_prime_x1 = df(x1)
        f_prime_x2 = df(x2)
        
        # Validar si el denominador es cero
        if abs(f_prime_x2 - f_prime_x1) < 1e-9:
            return (None, None, None)
        
        # Step 2: Calcular el nuevo punto z
        z = x2 - (f_prime_x2 * (x2 - x1)) / (f_prime_x2 - f_prime_x1)
        f_prime_z = df(z)
        
        # Plotear el nuevo punto en cada iteración
        ax.scatter(z, f(z), color='red', zorder=5)

        # Step 3: Condición de terminación
        if abs(f_prime_z) <= epsilon:
            # Marcar el punto final en la gráfica
            ax.scatter(z, f(z), color='green', s=100, zorder=6, label=f'Mínimo en x={z:.4f}')
            ax.legend()
            return (z, f(z), fig)
        
        # Actualizar los puntos para la próxima iteración
        x1 = x2
        x2 = z

        # Opcional: limitar el número de iteraciones para evitar bucles infinitos
        if abs(x2 - x1) < 1e-9:
             return (None, None, None)
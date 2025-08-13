import numpy as np
import matplotlib.pyplot as plt

def get_fibonacci_numbers(n):
    """Genera los primeros n números de Fibonacci."""
    fib = [0, 1]
    while len(fib) <= n:
        fib.append(fib[-1] + fib[-2])
    return fib

def fibonacci_search(f, a, b, n_evaluations):
    """
    Implementa el Método de Búsqueda de Fibonacci para encontrar el mínimo de 
    una función unimodal en un intervalo [a, b].

    Args:
        f (function): La función a minimizar.
        a (float): El límite inferior del intervalo.
        b (float): El límite superior del intervalo.
        n_evaluations (int): El número deseado de evaluaciones de la función.

    Returns:
        tuple: Una tupla que contiene:
               - El intervalo final [a, b] que acota el mínimo.
               - Una figura de Matplotlib con la gráfica del proceso.
    """
    # Step 1: Inicialización
    L = b - a
    fib = get_fibonacci_numbers(n_evaluations + 1)
    k = 2
    
    # Preparar la figura para la visualización del proceso
    x_values = np.linspace(a, b, 500)
    y_values = f(x_values)
    
    fig, ax = plt.subplots()
    ax.plot(x_values, y_values, label='f(x)', color='blue')
    ax.set_title(f'Método de Búsqueda de Fibonacci en [{a}, {b}]')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.grid(True)
    ax.legend()
    
    # Evaluar f(x1) y f(x2) una sola vez al principio si es necesario.
    # En la primera iteración, calculamos L_k y los puntos.
    L_k = (fib[n_evaluations - k + 1] / fib[n_evaluations + 1]) * L
    x1 = a + L_k
    x2 = b - L_k
    f_x1 = f(x1)
    f_x2 = f(x2)
    ax.scatter([x1, x2], [f_x1, f_x2], color='red', zorder=5)

    # Bucle principal del método
    while True:
        # Step 3: Aplicar la regla de eliminación de regiones
        if f_x1 < f_x2:
            # Eliminar la región [x2, b]
            b = x2
            x2 = x1
            f_x2 = f_x1
            
            L_k = (fib[n_evaluations - k] / fib[n_evaluations + 1]) * L
            x1 = a + L_k
            f_x1 = f(x1)
        elif f_x1 > f_x2:
            # Eliminar la región [a, x1]
            a = x1
            x1 = x2
            f_x1 = f_x2
            
            L_k = (fib[n_evaluations - k] / fib[n_evaluations + 1]) * L
            x2 = b - L_k
            f_x2 = f(x2)
        else:
            # En caso de igualdad, eliminar ambos extremos.
            a = x1
            b = x2
            L_k = (fib[n_evaluations - k] / fib[n_evaluations + 1]) * L
            x1 = a + L_k
            x2 = b - L_k
            f_x1 = f(x1)
            f_x2 = f(x2)

        # Plotear los nuevos puntos intermedios en cada iteración
        ax.scatter([x1, x2], [f_x1, f_x2], color='red', zorder=5)
        ax.axvline(a, color='gray', linestyle='--')
        ax.axvline(b, color='gray', linestyle='--')
            
        # Step 4: Condición de terminación
        if k == n_evaluations:
            # Marcar el intervalo final en la gráfica
            ax.axvline(a, color='green', linestyle='-', label=f'Intervalo Final [{a:.4f}, {b:.4f}]')
            ax.axvline(b, color='green', linestyle='-')
            ax.legend()
            return (a, b, fig)
        
        k += 1
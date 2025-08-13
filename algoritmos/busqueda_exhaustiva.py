import numpy as np
import matplotlib.pyplot as plt

def exhaustive_search(f, a, b, n):
    """
    Implementa el método de búsqueda exhaustiva para encontrar el mínimo
    de una función f en un intervalo [a, b] y genera una gráfica del proceso.

    Args:
        f (function): La función a minimizar.
        a (float): El límite inferior del intervalo.
        b (float): El límite superior del intervalo.
        n (int): El número de puntos intermedios.
    
    Returns:
        tuple: Una tupla que contiene:
               - El punto mínimo encontrado y el valor mínimo de la función.
               - Una figura de Matplotlib con la gráfica de la función y el mínimo.
               Devuelve (None, None, None) si no se encuentra un mínimo en el intervalo.
    """
    # Paso 1: Inicialización
    delta_x = (b - a) / (n + 1)
    x1 = a
    x2 = x1 + delta_x
    x3 = x2 + delta_x

    # Preparar datos para la gráfica
    x_values = np.linspace(a, b, 500)
    y_values = [f(x) for x in x_values]
    
    fig, ax = plt.subplots()
    ax.plot(x_values, y_values, label='f(x)', color='blue')
    ax.set_title(f'Búsqueda Exhaustiva en [{a}, {b}]')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.grid(True)
    ax.legend()
    
    # Marcadores para los puntos de búsqueda
    search_points = []
    
    while True:
        # Paso 2: Evaluación y condición de terminación
        f_x1 = f(x1)
        f_x2 = f(x2)
        f_x3 = f(x3)

        # Añadir los puntos al registro para la gráfica
        search_points.append(x1)
        
        if f_x1 >= f_x2 and f_x2 <= f_x3:
            # El mínimo local se encuentra en el intervalo (x1, x3).
            # Marcar el punto final encontrado
            ax.scatter(x2, f_x2, color='red', zorder=5, label=f'Mínimo en x={x2:.4f}')
            ax.legend()
            return (x2, f_x2, fig)
        
        # Si la condición no se cumple, actualizamos los puntos
        x1 = x2
        x2 = x3
        x3 = x3 + delta_x

        # Paso 3: Condición de continuación
        if x3 > b:
            # Si x3 excede el límite superior, no se encontró un mínimo
            # local que cumpla la condición. El mínimo puede ser a o b.
            f_a = f(a)
            f_b = f(b)
            
            if f_a < f_b:
                min_x = a
                min_val = f_a
            else:
                min_x = b
                min_val = f_b
                
            ax.scatter(min_x, min_val, color='orange', zorder=5, label=f'Mínimo en borde en x={min_x:.4f}')
            ax.legend()
            return (min_x, min_val, fig)

    # Si por alguna razón el bucle termina sin encontrar un mínimo
    return (None, None, None)
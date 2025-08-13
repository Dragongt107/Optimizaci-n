import numpy as np
import matplotlib.pyplot as plt

def region_elimination(f, a, b, precision=0.01):
    """
    Implementa el método de eliminación de regiones para encontrar un intervalo 
    que contiene el mínimo de una función unimodal en [a, b].

    Args:
        f (function): La función a minimizar.
        a (float): El límite inferior del intervalo.
        b (float): El límite superior del intervalo.
        precision (float): La tolerancia para la longitud del intervalo final.

    Returns:
        tuple: Una tupla que contiene:
               - El intervalo [a, b] acotado que contiene el mínimo.
               - Una figura de Matplotlib con la gráfica del proceso.
    """
    # Preparar la figura para la visualización del proceso
    x_values = np.linspace(a, b, 500)
    y_values = f(x_values)
    
    fig, ax = plt.subplots()
    ax.plot(x_values, y_values, label='f(x)', color='blue')
    ax.set_title(f'Eliminación de Regiones en [{a}, {b}]')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.grid(True)
    ax.legend()
    
    # Bucle principal del método
    while (b - a) > precision:
        # Calcular los puntos intermedios
        x1 = a + (b - a) / 3
        x2 = b - (b - a) / 3
        
        f_x1 = f(x1)
        f_x2 = f(x2)

        # Plotear los puntos intermedios en cada iteración
        ax.scatter([x1, x2], [f_x1, f_x2], color='red', zorder=5)

        # Criterio de eliminación de regiones
        if f_x1 > f_x2:
            # Eliminar la región [a, x1]
            a = x1
        elif f_x1 < f_x2:
            # Eliminar la región [x2, b]
            b = x2
        else: # f_x1 == f_x2
            # Eliminar las regiones [a, x1] y [x2, b]
            a = x1
            b = x2
            
        # Opcional: Actualizar las líneas del intervalo en la gráfica en cada iteración
        ax.axvline(a, color='gray', linestyle='--')
        ax.axvline(b, color='gray', linestyle='--')
            
    # Marcar el intervalo final en la gráfica
    ax.axvline(a, color='green', linestyle='-', label=f'Intervalo Final [{a:.4f}, {b:.4f}]')
    ax.axvline(b, color='green', linestyle='-')
    ax.legend()

    return (a, b, fig)
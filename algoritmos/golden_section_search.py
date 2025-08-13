import numpy as np
import matplotlib.pyplot as plt

def golden_section_search(f, a, b, epsilon=0.01):
    """
    Implementa el Método de Búsqueda de la Sección Dorada para encontrar el mínimo de 
    una función unimodal en un intervalo [a, b].

    Args:
        f (function): La función a minimizar.
        a (float): El límite inferior del intervalo.
        b (float): El límite superior del intervalo.
        epsilon (float): La tolerancia para la longitud del intervalo final.

    Returns:
        tuple: Una tupla que contiene:
               - El intervalo final [a, b] que acota el mínimo.
               - Una figura de Matplotlib con la gráfica del proceso.
    """
    # Step 1: Inicialización
    # No es necesario normalizar si trabajamos directamente con los límites a y b
    L = b - a
    
    # Preparar la figura para la visualización del proceso
    x_values = np.linspace(a, b, 500)
    y_values = f(x_values)
    
    fig, ax = plt.subplots()
    ax.plot(x_values, y_values, label='f(x)', color='blue')
    ax.set_title(f'Método de Búsqueda de la Sección Dorada en [{a}, {b}]')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.grid(True)
    ax.legend()
    
    # Step 2: Calcular los puntos iniciales
    phi = 0.6180339887  # La proporción áurea
    x1 = a + (1 - phi) * (b - a)
    x2 = a + phi * (b - a)
    
    f_x1 = f(x1)
    f_x2 = f(x2)
    
    ax.scatter([x1, x2], [f_x1, f_x2], color='red', zorder=5)

    # Bucle principal del método
    while (b - a) > epsilon:
        # Usar la regla de eliminación de regiones
        if f_x1 < f_x2:
            # Eliminar la región [x2, b]
            b = x2
            x2 = x1
            f_x2 = f_x1
            
            x1 = a + (1 - phi) * (b - a)
            f_x1 = f(x1)
        else: # f_x1 >= f_x2
            # Eliminar la región [a, x1]
            a = x1
            x1 = x2
            f_x1 = f_x2
            
            x2 = a + phi * (b - a)
            f_x2 = f(x2)
            
        # Plotear los nuevos puntos intermedios en cada iteración
        ax.scatter([x1, x2], [f_x1, f_x2], color='red', zorder=5)
        ax.axvline(a, color='gray', linestyle='--')
        ax.axvline(b, color='gray', linestyle='--')
            
    # Marcar el intervalo final en la gráfica
    ax.axvline(a, color='green', linestyle='-', label=f'Intervalo Final [{a:.4f}, {b:.4f}]')
    ax.axvline(b, color='green', linestyle='-')
    ax.legend()
    
    return (a, b, fig)
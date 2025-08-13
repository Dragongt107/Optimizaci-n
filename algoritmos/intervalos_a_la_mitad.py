import numpy as np
import matplotlib.pyplot as plt

def intervalos_a_la_mitad(f, a, b, epsilon=0.01):
    """
    Implementa el Método de Intervalos a la Mitad para encontrar el mínimo de 
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
    # Preparar la figura para la visualización del proceso
    x_values = np.linspace(a, b, 500)
    y_values = f(x_values)
    
    fig, ax = plt.subplots()
    ax.plot(x_values, y_values, label='f(x)', color='blue')
    ax.set_title(f'Método de Intervalos a la Mitad en [{a}, {b}]')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.grid(True)
    ax.legend()
    
    # Bucle principal del método
    L = b - a
    
    while L > epsilon:
        # Step 1: Calcular el punto medio
        xm = (a + b) / 2
        
        # Step 2: Calcular x1 y x2
        x1 = a + L / 4
        x2 = b - L / 4
        
        f_x1 = f(x1)
        f_x2 = f(x2)
        f_xm = f(xm)
        
        # Plotear los puntos intermedios en cada iteración
        ax.scatter([x1, x2, xm], [f_x1, f_x2, f_xm], color='red', zorder=5)

        # Step 3: Comparar f(x1) con f(xm)
        if f_x1 < f_xm:
            # Eliminar [xm, b]
            b = xm
            xm = x1
        else:
            # Step 4: Comparar f(x2) con f(xm)
            if f_x2 < f_xm:
                # Eliminar [a, xm]
                a = xm
                xm = x2
            else:
                # Eliminar [a, x1] y [x2, b]
                a = x1
                b = x2
        
        # Step 5: Calcular la nueva longitud y verificar la condición de terminación
        L = b - a
            
        # Opcional: Actualizar las líneas del intervalo en la gráfica en cada iteración
        ax.axvline(a, color='gray', linestyle='--')
        ax.axvline(b, color='gray', linestyle='--')
            
    # Marcar el intervalo final en la gráfica
    ax.axvline(a, color='green', linestyle='-', label=f'Intervalo Final [{a:.4f}, {b:.4f}]')
    ax.axvline(b, color='green', linestyle='-')
    ax.legend()
    
    return (a, b, fig)
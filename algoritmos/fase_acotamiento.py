import numpy as np
import matplotlib.pyplot as plt

def bounding_phase_method(f, x0, delta):
    """
    Implementa el Método de Fase de Acotamiento (Bounding Phase Method)
    para encontrar un intervalo que contenga el mínimo de una función f.

    Args:
        f (function): La función a minimizar.
        x0 (float): El punto inicial de adivinación.
        delta (float): El incremento inicial.
    
    Returns:
        tuple: Una tupla que contiene:
               - El intervalo [a, b] que acota el mínimo.
               - Una figura de Matplotlib con la gráfica del proceso.
               Devuelve (None, None, None) si el método falla.
    """
    # Paso 1: Elegir un punto inicial x0 y un incremento delta. Establecer k = 0.
    k = 0
    
    # Paso 2: Determinar la dirección del mínimo.
    if f(x0 - delta) >= f(x0) and f(x0) <= f(x0 + delta):
        # El mínimo se encuentra en el intervalo [x0 - delta, x0 + delta].
        # No se necesita más búsqueda, devolvemos este intervalo.
        a = x0 - delta
        b = x0 + delta
        
        x_values = np.linspace(a - abs(b-a)*0.2, b + abs(b-a)*0.2, 500)
        y_values = f(x_values)
        fig, ax = plt.subplots()
        ax.plot(x_values, y_values, label='f(x)', color='blue')
        ax.scatter([a, b], [f(a), f(b)], color='green', zorder=5, label='Fronteras del intervalo')
        ax.set_title(f'Fase de Acotamiento: Mínimo en [{a:.4f}, {b:.4f}]')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.grid(True)
        ax.legend()
        return (a, b, fig)

    elif f(x0 - delta) <= f(x0) and f(x0) >= f(x0 + delta):
        # El mínimo está fuera del intervalo inicial. El método falla o la función no es unimodal.
        return (None, None, None)
    
    elif f(x0 - delta) >= f(x0) and f(x0) >= f(x0 + delta):
        # El mínimo está a la derecha. delta es positivo.
        # Set delta to be positive
        delta = abs(delta)
    elif f(x0 - delta) <= f(x0) and f(x0) <= f(x0 + delta):
        # El mínimo está a la izquierda. delta es negativo.
        # Set delta to be negative
        delta = -abs(delta)
    else:
        # Si la función no es unimodal, el método podría fallar.
        return (None, None, None)

    # Inicializar puntos para la gráfica
    x_history = [x0]
    
    # Preparamos la figura para la visualización del proceso
    x_values = np.linspace(x0 - 5 * abs(delta), x0 + 5 * abs(delta), 500)
    y_values = f(x_values)
    
    fig, ax = plt.subplots()
    ax.plot(x_values, y_values, label='f(x)', color='blue')
    ax.set_title(f'Fase de Acotamiento (desde x0={x0})')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.grid(True)
    
    # Paso 3 y 4: Iterar para acotar el mínimo.
    x_k = x0
    x_k_plus_1 = x_k + delta
    
    while True:
        # Step 4: If f(x^(k+1)) < f(x^(k)), set k = k + 1 and go to Step 3.
        # Se establece el nuevo punto de búsqueda y se compara.
        x_k_plus_1 = x_k + (2**k) * delta
        
        # Evaluar la condición de continuación
        if f(x_k_plus_1) < f(x_k):
            # Aún se está moviendo hacia el mínimo.
            ax.scatter(x_k, f(x_k), color='gray', zorder=5)
            x_k = x_k_plus_1
            k += 1
            
        else:
            # Se ha pasado el mínimo, el intervalo se ha acotado.
            if delta > 0:
                a = x_k_plus_1 - (2**k) * delta
                b = x_k_plus_1
            else: # delta es negativo
                a = x_k_plus_1
                b = x_k_plus_1 - (2**k) * delta

            # Marcar el intervalo final en la gráfica
            ax.axvline(a, color='green', linestyle='--', label=f'Intervalo [{a:.4f}, {b:.4f}]')
            ax.axvline(b, color='green', linestyle='--')
            ax.scatter([a, b], [f(a), f(b)], color='green', zorder=5, label='Fronteras del intervalo')
            ax.legend()
            
            # Devolvemos el intervalo acotado
            return (a, b, fig)
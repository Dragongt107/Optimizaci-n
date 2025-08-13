import numpy as np
import matplotlib.pyplot as plt

def hill_climbing(f, x0, epsilon=0.01, max_iter=1000, step_size=0.1):
    """
    Implementa el Método de Hill Climbing para encontrar el mínimo de una función.

    Args:
        f (function): La función objetivo multivariable.
        x0 (np.array): La solución inicial.
        epsilon (float): Criterio de terminación (cambio en el valor de la función).
        max_iter (int): Número máximo de iteraciones.
        step_size (float): Tamaño del paso para la generación del vecino.

    Returns:
        tuple: Una tupla con el mejor punto encontrado, el valor de la función en ese punto,
               y una figura de Matplotlib con la gráfica del proceso.
    """
    # Step 1: Inicialización
    x_current = x0
    f_current = f(x_current)
    history = [x_current]
    
    # Preparar la figura para la visualización del proceso
    fig, ax = plt.subplots()
    ax.set_title('Método de Hill Climbing')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.grid(True)
    
    # ---- VISUALIZACIÓN DE CONTORNOS ----
    if x0.shape[0] == 2:
        x_range = np.linspace(x0[0] - 2, x0[0] + 2, 100)
        y_range = np.linspace(x0[1] - 2, x0[1] + 2, 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.array([[f(np.array([x, y])) for x in x_range] for y in y_range])
        ax.contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.7)
    # ------------------------------------

    # Bucle principal del método
    k = 0
    while k < max_iter:
        # Step 2: Generar un nuevo punto (vecino) aleatorio
        random_step = np.random.uniform(-step_size, step_size, size=x0.shape)
        x_k_plus_1 = x_current + random_step
        
        # Step 3: Evaluar y comparar
        f_k_plus_1 = f(x_k_plus_1)
        
        # Si el nuevo punto es mejor, actualizar la solución actual
        if f_k_plus_1 < f_current:
            x_current = x_k_plus_1
            f_current = f_k_plus_1
            
            # Guardar el nuevo punto para la visualización
            history.append(x_current)
        
        k += 1
        
        # Criterio de terminación adicional: si no hay mejora significativa
        if len(history) > 1 and abs(f(history[-1]) - f(history[-2])) < epsilon:
            break

    # Graficar el camino de la búsqueda en el espacio 2D
    history = np.array(history)
    if history.shape[1] == 2:
        ax.plot(history[:, 0], history[:, 1], 'o-', color='blue', alpha=0.5, label='Camino de Búsqueda')
        ax.scatter(x0[0], x0[1], color='red', s=100, label='Punto Inicial')
        ax.scatter(x_current[0], x_current[1], color='green', s=100, label='Mejor Punto Encontrado')
        ax.legend()
    
    return (x_current, f_current, fig)
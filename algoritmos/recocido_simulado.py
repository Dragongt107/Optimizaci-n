import numpy as np
import matplotlib.pyplot as plt

def recocido_simulado(f, x0, T0=100.0, alpha=0.95, w_size=100, max_iter=1000):
    """
    Implementa el Método de Recocido Simulado (Simulated Annealing) para encontrar
    el mínimo de una función multivariable.

    Args:
        f (function): La función objetivo multivariable.
        x0 (np.array): La solución inicial.
        T0 (float): La temperatura inicial.
        alpha (float): El factor de enfriamiento (alpha < 1).
        w_size (int): El tamaño del 'metropolis size' (número de iteraciones a una temperatura).
        max_iter (int): Número máximo de iteraciones para la condición de terminación.

    Returns:
        tuple: Una tupla con el mejor punto encontrado, el valor de la función en ese punto,
               y una figura de Matplotlib con la gráfica del proceso.
    """
    # Step 1: Set X <- Xs
    X_current = x0
    # Step 2: Set Best <- Xs
    Best = x0
    
    T_current = T0
    history = [X_current]
    
    # Bucle principal del método
    k = 0
    while k < max_iter:
        for _ in range(w_size):
            U = X_current + np.random.normal(0, T_current / T0, size=x0.shape)
            
            if f(U) < f(Best):
                Best = U
            
            Delta = f(U) - f(X_current)
            if np.exp(-Delta / T_current) >= np.random.uniform(0.0, 1.0):
                X_current = U
                history.append(X_current)
            
        T_current = alpha * T_current
        k += w_size

    # Convertir la lista history en un array de NumPy antes de graficar
    history = np.array(history)
    
    # Preparar la figura para la visualización del proceso
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title('Método de Recocido Simulado')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.grid(True)
    
    # ---- VISUALIZACIÓN DE CONTORNOS MEJORADA ----
    if x0.shape[0] == 2:
        # Aquí ya podemos usar el slicing de NumPy en el array 'history'
        # Aseguramos que haya suficientes puntos para calcular el rango
        if len(history) > 1:
            x_min, x_max = np.min(history[:, 0]) - 1, np.max(history[:, 0]) + 1
            y_min, y_max = np.min(history[:, 1]) - 1, np.max(history[:, 1]) + 1
        else: # Si el historial es muy corto, usamos un rango por defecto
            x_min, x_max = x0[0] - 5, x0[0] + 5
            y_min, y_max = x0[1] - 5, x0[1] + 5
        
        x_range = np.linspace(x_min, x_max, 200)
        y_range = np.linspace(y_min, y_max, 200)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.array([[f(np.array([x, y])) for x in x_range] for y in y_range])
        
        ax.contour(X, Y, Z, levels=50, cmap='viridis', alpha=0.7) 
    # ------------------------------------

    # Graficar el camino de la búsqueda en el espacio 2D
    if history.shape[1] == 2:
        ax.plot(history[:, 0], history[:, 1], 'o-', color='blue', alpha=0.5, label='Camino de Búsqueda')
        ax.scatter(x0[0], x0[1], color='red', s=100, label='Punto Inicial')
        ax.scatter(Best[0], Best[1], color='green', s=100, label='Mejor Punto Encontrado')
        ax.legend()
        
    return (Best, f(Best), fig)
# File: algoritmos/nelder_mead.py

import numpy as np
import matplotlib.pyplot as plt

def nelder_mead(f, x0, intervalos, delta=1.0, gamma=2.0, beta=0.5, epsilon=1e-4, max_iter=200):
    """
    Implementación del algoritmo Nelder-Mead basada en la descripción proporcionada.
    Asume funciones en 2D.
    
    Args:
    f: Función objetivo a minimizar (toma un array numpy de 2 elementos).
    x0: Punto inicial (array numpy de 2 elementos).
    intervalos: Lista de tuplas con los límites para cada variable, para graficar.
    delta: Tamaño inicial para generar el simplex.
    gamma: Parámetro de expansión (>1).
    beta: Parámetro de contracción (0 < beta < 1).
    epsilon: Parámetro de terminación.
    max_iter: Máximo de iteraciones para evitar bucles infinitos.
    
    Returns:
    x_min: Punto mínimo encontrado.
    f_min: Valor mínimo de la función.
    fig: Figura de matplotlib con el gráfico de contorno y el primer/último simplex.
    """
    n = 2  # Dimensión fija en 2 para este caso
    
    # Crear simplex inicial
    simplex_initial = [
        np.array(x0),
        np.array(x0) + delta * np.array([1.0, 0.0]),
        np.array(x0) + delta * np.array([0.0, 1.0])
    ]
    
    # Evaluar la función en los puntos del simplex inicial
    f_values_initial = [f(p) for p in simplex_initial]
    
    # Ordenar el simplex inicial: del mejor (menor f) al peor (mayor f)
    idx_initial = np.argsort(f_values_initial)
    simplex_initial = [simplex_initial[i] for i in idx_initial]
    f_values_initial = [f_values_initial[i] for i in idx_initial]
    
    # Crear simplex final (inicializar con el inicial)
    simplex = simplex_initial.copy()
    f_values = f_values_initial.copy()
    
    for iteration in range(max_iter):
        # Ordenar el simplex: del mejor (menor f) al peor (mayor f)
        idx = np.argsort(f_values)
        simplex = [simplex[i] for i in idx]
        f_values = [f_values[i] for i in idx]
        
        x_l = simplex[0]   # Mejor punto
        x_g = simplex[n-1] # Siguiente al peor
        x_h = simplex[n]   # Peor punto
        
        # Calcular el centroide (excluyendo el peor punto)
        x_c = np.mean(simplex[:-1], axis=0)
        
        # Calcular el punto reflejado
        x_r = 2 * x_c - x_h
        f_r = f(x_r)
        
        # Inicialmente, x_new es x_r
        x_new = x_r
        f_new = f_r
        
        if f_r < f_values[0]:  # Expansión si f_r < f(x_l)
            x_new = (1 + gamma) * x_c - gamma * x_h
            f_new = f(x_new)
        elif f_r >= f_values[-1]:  # Contracción interna si f_r >= f(x_h)
            x_new = (1 - beta) * x_c + beta * x_h
            f_new = f(x_new)
        elif f_values[-2] < f_r < f_values[-1]:  # Contracción externa si f(x_g) < f_r < f(x_h)
            x_new = (1 + beta) * x_c - beta * x_h
            f_new = f(x_new)
        # En caso contrario, se acepta la reflexión (x_new permanece como x_r)
        
        # Reemplazar el peor punto
        simplex[-1] = x_new
        f_values[-1] = f_new
        
        # Calcular el criterio de terminación
        f_c = f(x_c)
        std_dev = np.sqrt(np.sum((np.array(f_values) - f_c) ** 2) / (n + 1))
        if std_dev <= epsilon:
            break
    
    # Obtener el mejor punto final
    min_idx = np.argmin(f_values)
    x_min = simplex[min_idx]
    f_min = f_values[min_idx]
    
    # Crear la gráfica de contorno y el primer/último simplex
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Generar malla para contorno basada en intervalos
    x_low, x_high = intervalos[0]
    y_low, y_high = intervalos[1]
    x_grid = np.linspace(x_low, x_high, 100)
    y_grid = np.linspace(y_low, y_high, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.array([[f(np.array([x, y])) for x in x_grid] for y in y_grid])
    
    # Graficar contorno
    ax.contour(X, Y, Z, levels=50, cmap='viridis')
    
    # Graficar el simplex inicial (triángulo)
    ax.plot([simplex_initial[0][0], simplex_initial[1][0], simplex_initial[2][0], simplex_initial[0][0]],
            [simplex_initial[0][1], simplex_initial[1][1], simplex_initial[2][1], simplex_initial[0][1]],
            'b-', label='Simplex Inicial')
    
    # Graficar el simplex final (triángulo)
    ax.plot([simplex[0][0], simplex[1][0], simplex[2][0], simplex[0][0]],
            [simplex[0][1], simplex[1][1], simplex[2][1], simplex[0][1]],
            'r-', label='Simplex Final')
    
    # Marcar el punto mínimo final
    ax.scatter(x_min[0], x_min[1], color='green', s=50, label='Mínimo encontrado')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Nelder-Mead: Contorno y Simplex Inicial/Final')
    ax.legend()
    
    return x_min, f_min, fig
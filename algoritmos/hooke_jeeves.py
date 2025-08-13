import numpy as np
import matplotlib.pyplot as plt

def exploratory_move(f, x_c, deltas, history=None):
    if history is None:
        history = []
    x = x_c.copy()
    history.append(x.copy())  # Append starting point for this move
    for i in range(len(x)):
        f_curr = f(x)
        x_plus = x.copy()
        x_plus[i] += deltas[i]
        f_plus = f(x_plus)
        x_minus = x.copy()
        x_minus[i] -= deltas[i]
        f_minus = f(x_minus)
        min_f = min(f_curr, f_plus, f_minus)
        if min_f == f_plus:
            x = x_plus
            history.append(x.copy())
        elif min_f == f_minus:
            x = x_minus
            history.append(x.copy())
    success = not np.array_equal(x, x_c)
    return x, success, history

def hooke_jeeves(f, x0, intervalos, deltas=1.0, alpha=2.0, epsilon=1e-4, max_iter=200):
    """
    Implementación del algoritmo Hooke-Jeeves basada en la descripción proporcionada.
    Asume funciones en 2D.
    
    Args:
    f: Función objetivo a minimizar.
    x0: Punto inicial.
    intervalos: Límites para graficar.
    deltas: Tamaño inicial de paso para cada variable (float o array).
    alpha: Factor de reducción (>1).
    epsilon: Tolerancia para terminación.
    max_iter: Máximo de iteraciones.
    
    Returns:
    x_min: Punto mínimo encontrado.
    f_min: Valor mínimo.
    fig: Figura con contorno, path inicial y final con movimientos.
    """
    history = []
    
    x = x0.copy()
    history.append(x.copy())
    
    if isinstance(deltas, float):
        delta = np.full(len(x), deltas)
    else:
        delta = np.array(deltas)
    
    k = 0
    x_k_minus_1 = None
    
    while np.all(delta >= epsilon) and k < max_iter:
        # Step 2: Exploratory move
        y, success, history = exploratory_move(f, x, delta, history)
        if success:
            # Success
            if x_k_minus_1 is not None:
                while True:
                    # Step 4: Pattern move
                    p = y + (y - x_k_minus_1)
                    history.append(p.copy())  # Append pattern point
                    # Step 5: Exploratory around p
                    z, _, history = exploratory_move(f, p, delta, history)
                    # Step 6: Check improvement
                    if f(z) < f(y):
                        x_k_minus_1 = y
                        y = z
                        k += 1
                    else:
                        break
            x_k_minus_1 = x
            x = y
            k += 1
        else:
            # Failure, reduce delta
            delta = delta / alpha
    
    # Get min
    f_values = [f(p) for p in history]
    min_idx = np.argmin(f_values)
    x_min = history[min_idx]
    f_min = f_values[min_idx]
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Contour
    x_low, x_high = intervalos[0]
    y_low, y_high = intervalos[1]
    x_grid = np.linspace(x_low, x_high, 100)
    y_grid = np.linspace(y_low, y_high, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.array([[f(np.array([xx, yy])) for xx in x_grid] for yy in y_grid])
    
    ax.contour(X, Y, Z, levels=50, cmap='viridis')
    
    # Path with arrows
    history = np.array(history)
    ax.plot(history[:, 0], history[:, 1], 'b-', alpha=0.3)  # Faint line for path
    
    # Arrows for movements
    for i in range(1, len(history)):
        dx = history[i, 0] - history[i-1, 0]
        dy = history[i, 1] - history[i-1, 1]
        # Guess if pattern or exploratory: pattern if larger move
        if np.sqrt(dx*2 + dy*2) > np.mean(delta) * 1.5:  # Rough way to distinguish pattern (larger)
            color = 'red'  # Pattern
        else:
            color = 'black'  # Exploratory
        ax.arrow(history[i-1, 0], history[i-1, 1], dx, dy, head_width=0.1, head_length=0.1, fc=color, ec=color, alpha=0.7)
    
    # Initial and final
    ax.scatter(history[0, 0], history[0, 1], color='blue', s=100, label='Initial Point')
    ax.scatter(history[-1, 0], history[-1, 1], color='green', s=100, label='Final Point')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Hooke-Jeeves: Contorno, Movimientos Inicial y Final')
    ax.legend()
    
    return x_min, f_min, fig
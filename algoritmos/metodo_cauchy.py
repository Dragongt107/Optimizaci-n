import numpy as np
import math
import matplotlib.pyplot as plt

def golden_section_search(g, a=0.0, b=2.0, tol=1e-4):
    # Implementación de Golden Section Search basada en la diapositiva
    PHI = (math.sqrt(5) - 1) / 2  # Aproximadamente 0.618
    aw, bw = a, b
    Lw = bw - aw
    while Lw > tol:
        w1 = aw + PHI * Lw
        w2 = bw - PHI * Lw  # Nota: en estándar, w1 = a + (1-phi)L, w2 = a + phi L
        # Pero ajustando al código de diapositiva, asumiendo w2 = aw + PHI*Lw, w1 = aw + (1-PHI)*Lw
        # Para min, si g(w1) < g(w2), eliminar [w2, b], b = w2
        # Si g(w1) > g(w2), eliminar [a, w1], a = w1
        # Para consistencia, corrijo a estándar
        tau = PHI  # 0.618
        w1 = a + (1 - tau) * (b - a)
        w2 = a + tau * (b - a)
        if g(w1) < g(w2):
            b = w2
        else:
            a = w1
        Lw = b - a
    return (a + b) / 2

def metodo_cauchy(f, grad_f, x0, intervalos, epsilon1=1e-4, epsilon2=1e-4, M=100, max_alpha=2.0):
    """
    Implementación del método de Cauchy (Steepest Descent) basado en la descripción proporcionada.
    Asume funciones en 2D.
    
    Args:
    f: Función objetivo a minimizar.
    grad_f: Función que calcula el gradiente.
    x0: Punto inicial.
    intervalos: Límites para graficar.
    epsilon1: Tolerancia para norma del gradiente y cambio relativo en x.
    epsilon2: Tolerancia para la búsqueda lineal.
    M: Máximo de iteraciones.
    max_alpha: Límite superior para alpha en la búsqueda lineal.
    
    Returns:
    x_min: Punto mínimo encontrado.
    f_min: Valor mínimo.
    fig: Figura con contorno y trayectoria de movimientos.
    """
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    
    k = 0
    while k < M:
        grad = grad_f(x)
        norm_grad = np.linalg.norm(grad)
        if norm_grad < epsilon1:
            break
        
        # Dirección de descenso: s = -grad
        s = -grad
        
        # Función para búsqueda lineal: g(alpha) = f(x + alpha * s)
        def g(alpha):
            return f(x + alpha * s)
        
        # Realizar Golden Section Search para encontrar alpha
        alpha = golden_section_search(g, a=0.0, b=max_alpha, tol=epsilon2)
        
        # Actualizar x
        x_new = x + alpha * s
        history.append(x_new.copy())
        
        # Criterio de terminación adicional: cambio relativo en x
        norm_diff = np.linalg.norm(x_new - x)
        norm_x = np.linalg.norm(x)
        if norm_x > 0 and norm_diff / norm_x < epsilon1:
            break
        
        x = x_new
        k += 1
    
    f_min = f(x)
    x_min = x
    
    # Crear gráfica
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Contorno
    x_low, x_high = intervalos[0]
    y_low, y_high = intervalos[1]
    x_grid = np.linspace(x_low, x_high, 100)
    y_grid = np.linspace(y_low, y_high, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.array([[f(np.array([xx, yy])) for xx in x_grid] for yy in y_grid])
    
    ax.contour(X, Y, Z, levels=50, cmap='viridis')
    
    # Trayectoria con flechas
    history = np.array(history)
    ax.plot(history[:, 0], history[:, 1], 'b-', alpha=0.3)
    for i in range(1, len(history)):
        dx = history[i, 0] - history[i-1, 0]
        dy = history[i, 1] - history[i-1, 1]
        ax.arrow(history[i-1, 0], history[i-1, 1], dx, dy, head_width=0.1, head_length=0.1, fc='black', ec='black', alpha=0.7)
    
    # Punto inicial y final
    ax.scatter(history[0, 0], history[0, 1], color='blue', s=100, label='Punto Inicial')
    ax.scatter(history[-1, 0], history[-1, 1], color='green', s=100, label='Punto Final')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Método de Cauchy: Contorno y Trayectoria de Descenso')
    ax.legend()
    
    return x_min, f_min, fig
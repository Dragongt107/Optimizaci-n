import numpy as np
import math
import matplotlib.pyplot as plt

def numerical_hessian(f, x, delta=1e-5):
    n = len(x)
    H = np.zeros((n, n))
    fx = f(x)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = delta
        fxp = f(x + ei)
        fxm = f(x - ei)
        # Diagonal
        H[i, i] = (fxp - 2*fx + fxm) / delta**2
        for j in range(i+1, n):
            ej = np.zeros(n)
            ej[j] = delta
            fxpjp = f(x + ei + ej)
            fxpjm = f(x + ei - ej)
            fxmjp = f(x - ei + ej)
            fxmjm = f(x - ei - ej)
            H[i, j] = (fxpjp - fxpjm - fxmjp + fxmjm) / (4 * delta**2)
            H[j, i] = H[i, j]
    return H

def golden_section_search(g, a=0.0, b=2.0, tol=1e-4):
    PHI = (math.sqrt(5) - 1) / 2  # Aproximadamente 0.618
    tau = PHI
    while b - a > tol:
        w1 = a + (1 - tau) * (b - a)
        w2 = a + tau * (b - a)
        if g(w1) < g(w2):
            b = w2
        else:
            a = w1
    return (a + b) / 2

def metodo_newton(f, grad_f, x0, intervalos, epsilon1=1e-4, epsilon2=1e-4, delta_hess=1e-5, M=100, max_alpha=2.0):
    """
    Implementación del método de Newton basado en la descripción proporcionada.
    Asume funciones en 2D.
    
    Args:
    f: Función objetivo a minimizar.
    grad_f: Función que calcula el gradiente.
    x0: Punto inicial.
    intervalos: Límites para graficar.
    epsilon1: Tolerancia para norma del gradiente y cambio relativo en x.
    epsilon2: Tolerancia para la búsqueda lineal.
    delta_hess: Delta para cómputo numérico de la Hessiana.
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
        
        # Computar Hessiana numéricamente
        H = numerical_hessian(f, x, delta=delta_hess)
        
        # Dirección: s = - H^{-1} grad
        try:
            inv_H = np.linalg.inv(H)
            s = -inv_H @ grad
        except np.linalg.LinAlgError:
            # Si Hessiana no invertible, usar dirección de gradiente como fallback
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
    ax.set_title('Método de Newton: Contorno y Trayectoria de Descenso')
    ax.legend()
    
    return x_min, f_min, fig
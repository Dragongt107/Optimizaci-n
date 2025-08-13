import numpy as np
import matplotlib.pyplot as plt

def caminata_aleatoria(f, x0, epsilon=0.01, max_iter=1000, step_size=0.1):
    # Step 1: Inicialización
    x_best = x0
    f_best = f(x_best)
    history = [x_best]
    
    # Preparar la figura para la visualización del proceso
    fig, ax = plt.subplots()
    ax.set_title('Método de Caminata Aleatoria')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.grid(True)
    
    # ---- NUEVA ADICIÓN: VISUALIZACIÓN DE CONTORNOS ----
    if x0.shape[0] == 2:
        x_range = np.linspace(x0[0] - 2, x0[0] + 2, 100)
        y_range = np.linspace(x0[1] - 2, x0[1] + 2, 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.array([[f(np.array([x, y])) for x in x_range] for y in y_range])
        ax.contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.7)
    # ----------------------------------------------------

    # Bucle principal del método
    k = 0
    while k < max_iter:
        random_step = np.random.uniform(-step_size, step_size, size=x0.shape)
        x_k_plus_1 = x_best + random_step
        
        f_k_plus_1 = f(x_k_plus_1)
        
        if f_k_plus_1 < f_best:
            x_best = x_k_plus_1
            f_best = f_k_plus_1
            history.append(x_best)
        
        k += 1
        
        if len(history) > 10 and abs(f(history[-1]) - f(history[-2])) < epsilon:
            break

    # Graficar el camino de la búsqueda
    history = np.array(history)
    if history.shape[1] == 2:
        ax.plot(history[:, 0], history[:, 1], 'o-', color='blue', alpha=0.5, label='Camino de Búsqueda')
        ax.scatter(x0[0], x0[1], color='red', s=100, label='Punto Inicial')
        ax.scatter(x_best[0], x_best[1], color='green', s=100, label='Mejor Punto Encontrado')
        ax.legend()
    
    return (x_best, f_best, fig)
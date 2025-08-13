import numpy as np
import matplotlib.pyplot as plt

def newton_raphson(f, df, ddf, x0, epsilon=0.01):
    """
    Implementa el Método de Newton-Raphson para encontrar el mínimo de 
    una función f, dadas su primera y segunda derivada.

    Args:
        f (function): La función a minimizar.
        df (function): La primera derivada de la función.
        ddf (function): La segunda derivada de la función.
        x0 (float): La suposición inicial.
        epsilon (float): La tolerancia para la condición de terminación.

    Returns:
        tuple: Una tupla que contiene:
               - El punto mínimo encontrado.
               - El valor de la función en ese punto.
               - Una figura de Matplotlib con la gráfica del proceso.
               Devuelve (None, None, None) si el método no converge.
    """
    # Step 1: Inicialización
    x_k = x0
    k = 1
    
    # Preparar la figura para la visualización del proceso
    x_range = np.linspace(x0 - 5, x0 + 5, 500)
    y_values = f(x_range)
    
    fig, ax = plt.subplots()
    ax.plot(x_range, y_values, label='f(x)', color='blue')
    ax.set_title('Método de Newton-Raphson')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.grid(True)
    ax.legend()
    
    while True:
        # Step 2: Calcular la segunda derivada en el punto actual
        f_ddf_xk = ddf(x_k)
        
        # Validar si la segunda derivada es cercana a cero
        if abs(f_ddf_xk) < 1e-9:
            return (None, None, None)

        # Step 3: Calcular el nuevo punto
        f_df_xk = df(x_k)
        x_k_plus_1 = x_k - (f_df_xk / f_ddf_xk)
        
        # Plotear el punto de la iteración actual
        ax.scatter(x_k_plus_1, f(x_k_plus_1), color='red', zorder=5)

        # Step 4: Condición de terminación
        if abs(df(x_k_plus_1)) < epsilon:
            # Marcar el mínimo final en la gráfica
            ax.scatter(x_k_plus_1, f(x_k_plus_1), color='green', s=100, zorder=6, label=f'Mínimo en x={x_k_plus_1:.4f}')
            ax.legend()
            return (x_k_plus_1, f(x_k_plus_1), fig)
        
        x_k = x_k_plus_1
        k += 1
        
        # Opcional: limitar el número de iteraciones para evitar bucles infinitos
        if k > 100:
            return (None, None, None)
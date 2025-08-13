import numpy as np
import matplotlib.pyplot as plt

def metodo_biseccion(f, df, a, b, epsilon=0.01):
    """
    Implementa el Método de Bisección para encontrar un punto donde df(x) = 0,
    que corresponde a un punto crítico de la función f.

    Args:
        f (function): La función original.
        df (function): La primera derivada de la función.
        a (float): El límite inferior del intervalo.
        b (float): El límite superior del intervalo.
        epsilon (float): La tolerancia para la condición de terminación.

    Returns:
        tuple: Una tupla que contiene:
               - El punto crítico encontrado (donde df(x) ≈ 0).
               - El valor de la función en ese punto.
               - Una figura de Matplotlib con la gráfica del proceso.
               Devuelve (None, None, None) si el método no converge.
    """
    # Step 1: Inicialización
    x1 = a
    x2 = b
    
    # Preparar la figura para la visualización del proceso
    x_values = np.linspace(a - abs(b-a)*0.2, b + abs(b-a)*0.2, 500)
    y_values = f(x_values)
    
    fig, ax = plt.subplots()
    ax.plot(x_values, y_values, label='f(x)', color='blue')
    ax.set_title('Método de Bisección')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.grid(True)
    ax.legend()
    
    # Validar que f'(a) y f'(b) tengan signos opuestos
    if np.sign(df(a)) == np.sign(df(b)):
        return (None, None, None)

    # Bucle principal del método
    while True:
        # Step 2: Calcular el punto medio z y evaluar f'(z)
        z = (x1 + x2) / 2
        f_prime_z = df(z)

        # Plotear el punto medio en cada iteración
        ax.scatter(z, f(z), color='red', zorder=5)

        # Step 3: Condición de terminación
        if abs(f_prime_z) <= epsilon:
            # Marcar el punto final en la gráfica
            ax.scatter(z, f(z), color='green', s=100, zorder=6, label=f'Mínimo en x={z:.4f}')
            ax.legend()
            return (z, f(z), fig)
        
        # Eliminar una de las mitades
        if np.sign(f_prime_z) == np.sign(df(x1)):
            x1 = z
        else:
            x2 = z

        # Opcional: limitar el número de iteraciones para evitar bucles infinitos
        if abs(x2 - x1) < 1e-9:
             return (None, None, None)
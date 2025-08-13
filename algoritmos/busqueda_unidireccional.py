import numpy as np
import matplotlib.pyplot as plt

# Importar un método de optimización univariable.
# Usaremos la Búsqueda Exhaustiva como ejemplo, pero podrías usar otro.
from .busqueda_exhaustiva import exhaustive_search

def unidireccional_search(f_multivariable, x_t, s_t, univariable_method, a_range, n_points):
    """
    Implementa el Método de Búsqueda Unidireccional.

    Args:
        f_multivariable (function): La función objetivo multivariable.
        x_t (np.array): El punto de inicio de la búsqueda.
        s_t (np.array): La dirección de búsqueda.
        univariable_method (function): El método de optimización univariable a usar.
        a_range (tuple): El rango para la variable alpha (e.g., (0, 1)).
        n_points (int): Número de puntos para el método univariable (ej. Búsqueda Exhaustiva).

    Returns:
        tuple: Una tupla con el punto óptimo encontrado (x_opt) y una figura.
    """
    # 1. Definir la función univariable en términos de alpha (α)
    # Cualquier punto sobre la línea puede ser expresado como x(α) = x^(t) + αs^(t)
    def f_univariable(alpha):
        x_alpha = x_t + alpha * s_t
        return f_multivariable(x_alpha)

    # 2. Utilizar un método de optimización univariable para encontrar el mejor alpha
    alpha_min, _, fig_alpha = univariable_method(f_univariable, a_range[0], a_range[1], n_points)

    if alpha_min is None:
        return (None, None)

    # 3. Calcular el punto óptimo final x(α)
    x_opt = x_t + alpha_min * s_t
    
    # 4. Generar la visualización del proceso en el espacio multivariable
    
    # Crear una figura para mostrar el camino de la búsqueda
    fig_multivariable, ax_multivariable = plt.subplots()
    
    # Graficar la función multivariable (contornos)
    x1_min, x1_max = x_t[0] - 5, x_t[0] + 5
    x2_min, x2_max = x_t[1] - 5, x_t[1] + 5

    x1 = np.linspace(x1_min, x1_max, 100)
    x2 = np.linspace(x2_min, x2_max, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.array([[f_multivariable(np.array([x, y])) for x in x1] for y in x2])

    ax_multivariable.contour(X1, X2, Z, levels=15)
    ax_multivariable.set_title('Búsqueda Unidireccional')
    ax_multivariable.set_xlabel('x1')
    ax_multivariable.set_ylabel('x2')
    ax_multivariable.grid(True)

    # Trazar la línea de búsqueda
    alphas = np.linspace(a_range[0], a_range[1], 100)
    line_points = np.array([x_t + alpha * s_t for alpha in alphas])
    ax_multivariable.plot(line_points[:, 0], line_points[:, 1], 'r--', label='Dirección de Búsqueda')

    # Marcar el punto inicial y el punto óptimo
    ax_multivariable.scatter(x_t[0], x_t[1], color='blue', zorder=5, label='Punto Inicial ($x^{(t)}$)')
    ax_multivariable.scatter(x_opt[0], x_opt[1], color='green', zorder=5, s=100, label='Punto Óptimo ($x(α)$)')
    ax_multivariable.legend()

    return (x_opt, fig_multivariable)
import streamlit as st
import pandas as pd
import numpy as np


from algoritmos.busqueda_exhaustiva import exhaustive_search
from algoritmos.fase_acotamiento import bounding_phase_method
from algoritmos.region_eliminacion import region_elimination
from algoritmos.intervalos_a_la_mitad import intervalos_a_la_mitad
from algoritmos.fibonacci_search import fibonacci_search
from algoritmos.golden_section_search import golden_section_search
from algoritmos.newton_raphson import newton_raphson
from algoritmos.metodo_biseccion import metodo_biseccion
from algoritmos.metodo_secante import metodo_secante
from algoritmos.busqueda_unidireccional import unidireccional_search
from algoritmos.caminata_aleatoria import caminata_aleatoria
from algoritmos.hill_climbing import hill_climbing
from algoritmos.recocido_simulado import recocido_simulado
from algoritmos.nelder_mead import nelder_mead
from algoritmos.hooke_jeeves import hooke_jeeves
from algoritmos.metodo_cauchy import metodo_cauchy
from algoritmos.metodo_newton import metodo_newton

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Optimizacion",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DATOS: FUNCIONES Y ALGORITMOS ---
FUNCIONES_UNIVARIADAS = {
    "Función 1": {
        "intervalos": [(0.001, 10, lambda x: x**2 + 54/x)],
        "latex": r'''f(x) = x^2 + \frac{54}{x}''',
        "latex_derivada": r'''f'(x) = 2x - \frac{54}{x^2}''',
        "dominio": "(0, 10]",
        "df": lambda x: 2*x - 54/(x**2),
        "ddf": lambda x: 2 + 108/(x**3)
    },
    "Función 2": {
        "intervalos": [(0, 5, lambda x: x**3 + 2*x - 3)],
        "latex": r'''f(x) = x^3 + 2x - 3''',
        "latex_derivada": r'''f'(x) = 3x^2 + 2''',
        "dominio": "(0, 5]"
    },
    "Función 3": {
        "intervalos": [(-2.5, 2.5, lambda x: x**4 + x**2 - 33)],
        "latex": r'''f(x) = x^4 + x^2 - 33''',
        "latex_derivada": r'''f'(x) = 4x^3 + 2x''',
        "dominio": "[-2.5, 2.5]"
    },
    "Función 4": {
        "intervalos": [(-1.5, 3, lambda x: 3*x**4 - 8*x**3 - 6*x**2 + 12*x)],
        "latex": r'''f(x) = 3x^4 - 8x^3 - 6x^2 + 12x''',
        "latex_derivada": r'''f'(x) = 12x^3 - 24x^2 - 12x + 12''',
        "dominio": "[-1.5, 3]"
    },
    "Función Lata": {
        "intervalos": [(0.001, 10, lambda x: 2*np.pi*x*x + (500 / x))],
        "latex": r'''f(x) = 2\pi r^2 + \frac{500}{r}''',
        "latex_derivada": r'''f'(x) = 4\pi r - \frac{500}{r^2}''',
        "dominio": "(0, ∞)"
    },
    "Función Caja": {
        "intervalos": [(0.1, 10, lambda x: 4*x**3 - 60*x**2 + 200*x)],
        "latex": r'''f(x) = 4x^3 - 60x^2 + 200x''',
        "latex_derivada": r'''f'(x) = 12x^2 - 120x + 200''',
        "dominio": "(0, ∞)"
    }
}

FUNCIONES_MULTIVARIADAS = {
    "Rastrigin": {
        "funcion": lambda x: 10 * 2 + (x[0]**2 - 10 * np.cos(2 * np.pi * x[0])) + (x[1]**2 - 10 * np.cos(2 * np.pi * x[1])),
        "gradiente": lambda x: np.array([2*x[0] + 20*np.pi*np.sin(2*np.pi*x[0]), 2*x[1] + 20*np.pi*np.sin(2*np.pi*x[1])]),
        "intervalos": [(-5.12, 5.12), (-5.12, 5.12)],
        "latex": r"f(x, y) = 20 + (x^2 - 10 \cos(2\pi x)) + (y^2 - 10 \cos(2\pi y))",
        "dominio": "[-5.12, 5.12] x [-5.12, 5.12]",
        "minimo": "f(0, 0) = 0",
        "minimo_coords": np.array([0.0, 0.0])
    },
    "Ackley": {
        "funcion": lambda x: -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0]**2 + x[1]**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))) + np.e + 20,
        "gradiente": lambda x: np.array([
            (2 * x[0] * np.exp(-0.2 * np.sqrt(0.5 * (x[0]**2 + x[1]**2)))) / (np.sqrt(0.5 * (x[0]**2 + x[1]**2)) + 1e-9) + np.pi * np.sin(2 * np.pi * x[0]) * np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))),
            (2 * x[1] * np.exp(-0.2 * np.sqrt(0.5 * (x[0]**2 + x[1]**2)))) / (np.sqrt(0.5 * (x[0]**2 + x[1]**2)) + 1e-9) + np.pi * np.sin(2 * np.pi * x[1]) * np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1])))
        ]),
        "intervalos": [(-5, 5), (-5, 5)],
        "latex": r"f(x,y) = -20e^{-0.2\sqrt{0.5(x^2+y^2)}} - e^{0.5(\cos(2\pi x)+\cos(2\pi y))} + e + 20",
        "dominio": "[-5, 5] x [-5, 5]",
        "minimo": "f(0, 0) = 0",
        "minimo_coords": np.array([0.0, 0.0])
    },
    "Sphere": {
        "funcion": lambda x: x[0]**2 + x[1]**2,
        "gradiente": lambda x: np.array([2*x[0], 2*x[1]]),
        "intervalos": [(-10, 10), (-10, 10)],
        "latex": r"f(x, y) = x^2 + y^2",
        "dominio": "[-10, 10] x [-10, 10]",
        "minimo": "f(0, 0) = 0",
        "minimo_coords": np.array([0.0, 0.0])
    },
    "Rosenbrock": {
        "funcion": lambda x: 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2,
        "gradiente": lambda x: np.array([-400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0]), 200*(x[1]-x[0]**2)]),
        "intervalos": [(-2, 2), (-1, 3)],
        "latex": r"f(x, y) = 100(y - x^2)^2 + (1 - x)^2",
        "dominio": "[-2, 2] x [-1, 3]",
        "minimo": "f(1, 1) = 0",
        "minimo_coords": np.array([1.0, 1.0])
    },
    "Beale": {
        "funcion": lambda x: (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2,
        "gradiente": lambda x: np.array([2*(1.5 - x[0] + x[0]*x[1])*(-1 + x[1]) + 2*(2.25 - x[0] + x[0]*x[1]**2)*(-1 + x[1]**2) + 2*(2.625 - x[0] + x[0]*x[1]**3)*(-1 + x[1]**3), 2*(1.5 - x[0] + x[0]*x[1])*x[0] + 2*(2.25 - x[0] + x[0]*x[1]**2)*(2*x[0]*x[1]) + 2*(2.625 - x[0] + x[0]*x[1]**3)*(3*x[0]*x[1]**2)]),
        "intervalos": [(-4.5, 4.5), (-4.5, 4.5)],
        "latex": r"f(x,y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2",
        "dominio": "[-4.5, 4.5] x [-4.5, 4.5]",
        "minimo": "f(3, 0.5) = 0",
        "minimo_coords": np.array([3.0, 0.5])
    },
    "Booth": {
        "funcion": lambda x: (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2,
        "gradiente": lambda x: np.array([2*(x[0] + 2*x[1] - 7) + 4*(2*x[0] + x[1] - 5), 4*(x[0] + 2*x[1] - 7) + 2*(2*x[0] + x[1] - 5)]),
        "intervalos": [(-10, 10), (-10, 10)],
        "latex": r"f(x, y) = (x + 2y - 7)^2 + (2x + y - 5)^2",
        "dominio": "[-10, 10] x [-10, 10]",
        "minimo": "f(1, 3) = 0",
        "minimo_coords": np.array([1.0, 3.0])
    },
    "Himmelblau": {
        "funcion": lambda x: (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2,
        "gradiente": lambda x: np.array([4*x[0]*(x[0]**2 + x[1] - 11) + 2*(x[0] + x[1]**2 - 7), 2*(x[0]**2 + x[1] - 11) + 4*x[1]*(x[0] + x[1]**2 - 7)]),
        "intervalos": [(-5, 5), (-5, 5)],
        "latex": r"f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2",
        "dominio": "[-5, 5] x [-5, 5]",
        "minimo": "Varios mínimos, ej: f(3, 2) = 0",
        "minimo_coords": np.array([3.0, 2.0])
    },
    "McCormick": {
        "funcion": lambda x: np.sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5*x[0] + 2.5*x[1] + 1,
        "gradiente": lambda x: np.array([np.cos(x[0] + x[1]) + 2*(x[0] - x[1]) - 1.5, np.cos(x[0] + x[1]) - 2*(x[0] - x[1]) + 2.5]),
        "intervalos": [(-1.5, 4), (-3, 4)],
        "latex": r"f(x, y) = \sin(x + y) + (x - y)^2 - 1.5x + 2.5y + 1",
        "dominio": "[-1.5, 4] x [-3, 4]",
        "minimo": "f(-0.547, -1.547) ≈ -1.913",
        "minimo_coords": np.array([-0.547, -1.547])
    }
}


ALGORITMOS_UNIVARIADOS = [
    'Búsqueda Exhaustiva', 'Fase de Acotamiento', 'Región de Eliminaciones',
    'Intervalos a la Mitad', 'Fibonacci Search', 'Golden Section Search',
    'Newton-Raphson', 'Método de Bisección', 'Método Secante'
]

ALGORITMOS_MULTIVARIADOS = [
    'Búsqueda Unidireccional', 'Caminata Aleatoria', 'Hill Climbing',
    'Recocido Simulado', 'Nelder-Mead', 'Método de Hooke-Jeeves',
    'Método del Gradiente', 'Método de Cauchy', 'Método de Newton'
]

# --- BARRA LATERAL (SIDEBAR) ---
st.sidebar.title("Menú de Navegación")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Selecciona una sección:",
    ["📚 Conceptos Teóricos", "⚙️ Visualizador de Algoritmos"]
)
st.sidebar.markdown("---")


# --- CONTENIDO PRINCIPAL ---
if page == "📚 Conceptos Teóricos":
    st.title("📈 Optimización: Un Enfoque Práctico")
    st.markdown("---")

    page_theory = st.radio("Secciones Teóricas", ["Conceptos Fundamentales", "Métodos de Optimización", "Problemas de Muestra"], horizontal=True)

    if page_theory == "Conceptos Fundamentales":
        st.header("¿Qué es la Optimización?")
        st.info("La optimización es el proceso de encontrar la mejor solución posible a un problema, ajustando el valor de las variables que pueden ser controladas, a veces bajo ciertas restricciones. Cualquier problema que requiera una decisión puede ser formulado como un problema de optimización.")
        
        st.subheader("Componentes de un Problema de Optimización")
        with st.expander("Ver componentes"):
            st.markdown("""
            Un problema de optimización se compone de:
            - **Variables de decisión**: Las variables que describen el sistema y cuyos valores se modifican en busca del óptimo.
            - **Función objetivo**: La métrica que determina qué tan buena es una solución, la cual puede ser minimizada o maximizada.
            - **Restricciones**: Las condiciones o limitaciones que las variables deben satisfacer. La región donde se cumplen todas las restricciones se llama **región factible**.
            """)

    elif page_theory == "Métodos de Optimización":
        st.header("Formas de Abordar un Problema de Optimización")
        st.markdown("Existen diferentes métodos, cada uno con sus ventajas y limitaciones:")

        st.subheader("1. Métodos Analíticos")
        with st.expander("Más información sobre los Métodos Analíticos"):
            st.markdown("""
            Basados en el cálculo diferencial, estos métodos determinan el mínimo o máximo al encontrar los puntos donde la primera derivada es cero.
            - **Pros**: No requieren el uso de una computadora.
            - **Contras**: No se pueden aplicar a problemas altamente no lineales o con más de dos o tres variables independientes.
            """)

        st.subheader("2. Métodos Gráficos")
        with st.expander("Más información sobre los Métodos Gráficos"):
            st.markdown("""
            Requieren la visualización de la función a optimizar.
            - **Pros**: Intuitivos para problemas con pocas variables.
            - **Contras**: Tienen una utilidad limitada, ya que la mayoría de los problemas prácticos tienen más de dos variables.
            """)
        
        st.subheader("3. Métodos Experimentales")
        with st.expander("Más información sobre los Métodos Experimentales"):
            st.markdown("""
            Ajustan las variables una por una y evalúan el criterio de optimización en cada paso.
            - **Pros**: Pueden acercarse al óptimo.
            - **Contras**: Pueden fallar al no considerar la interacción simultánea de varias variables.
            """)

        st.subheader("4. Métodos Numéricos")
        with st.expander("Más información sobre los Métodos Numéricos"):
            st.markdown("""
            El método más importante y general, que utiliza procedimientos iterativos para mejorar progresivamente una solución inicial.
            - **Pros**: Pueden resolver problemas complejos y son fáciles de programar en una computadora.
            - **Contras**: El proceso termina cuando se alcanza un criterio de convergencia, no siempre el óptimo global.
            """)

    elif page_theory == "Problemas de Muestra":
        st.header("Ejercicios de Clase")
        with st.expander("Problema: Optimización de la caja sin tapa"):
            st.markdown("""
            **Problema**: Se quiere construir una caja sin tapa a partir de una hoja de cartón de 20x10cm, cortando un cuadrado de lado L en cada esquina. El objetivo es determinar las dimensiones para que el volumen sea máximo, con la restricción de que `2 ≤ L ≤ 3 cm`.

            Para resolver esto se requiere:
            1. Escribir el problema en formato NLP (Programación No Lineal).
            2. Resolverlo analíticamente.
            3. Generar la gráfica del problema y verificar la respuesta.
            """)

elif page == "⚙️ Visualizador de Algoritmos":
    st.title("⚡ Visualizador de Algoritmos de Optimización")
    st.markdown("---")

    # --- BARRA LATERAL PARA ALGORITMOS ---
    with st.sidebar:
        st.markdown("---")
        st.markdown("<h2 style='text-align: center;'>Configuración de Algoritmos</h2>", unsafe_allow_html=True)
        st.divider()

        tipo_algoritmo = st.radio(
            "Tipo de Función",
            ('Univariada', 'Multivariada'),
            horizontal=True,
            key='tipo_funcion'
        )

        es_multivariado = (tipo_algoritmo == 'Multivariada')

        if es_multivariado:
            ALGORITMOS = ALGORITMOS_MULTIVARIADOS
            FUNCIONES = FUNCIONES_MULTIVARIADAS
        else:
            ALGORITMOS = ALGORITMOS_UNIVARIADOS
            FUNCIONES = FUNCIONES_UNIVARIADAS

        algoritmo_seleccionado = st.selectbox(
            'Selecciona un algoritmo:',
            ALGORITMOS,
            help="Elige un algoritmo de optimización para ejecutar."
        )

        funcion_seleccionada = st.selectbox(
            'Selecciona una función:',
            list(FUNCIONES.keys()),
            help="Elige una función para optimizar."
        )
        st.divider()
        st.info("Esta sección visualiza y ejecuta diversos algoritmos de optimización numérica.")

    # --- PANTALLA PRINCIPAL DE ALGORITMOS ---
    info_funcion = FUNCIONES[funcion_seleccionada]

    tab1, tab2 = st.tabs(["⚙️ Ejecución y Resultados", "📄 Detalles de la Función"])

    with tab1:
        st.header(f"Algoritmo: {algoritmo_seleccionado}", divider='orange')
        st.subheader(f"Función a optimizar: {funcion_seleccionada}")

        results_container = st.container(border=True)

        if algoritmo_seleccionado == 'Búsqueda Exhaustiva':
            with st.expander("Parámetros del Algoritmo"):
                n = st.slider('Número de puntos (n)', 10, 1000, 100, 10)
            if st.button("Ejecutar", type="primary", use_container_width=True):
                a, b, f = info_funcion['intervalos'][0]
                min_point, min_value, fig = exhaustive_search(f, a, b, n)
                with results_container:
                    st.success(f"Punto mínimo encontrado en x ≈ {min_point:.6f} con valor f(x) ≈ {min_value:.6f}")
                    st.pyplot(fig)
        
        elif algoritmo_seleccionado == 'Fase de Acotamiento':
            with st.expander("Parámetros del Algoritmo"):
                x0 = st.number_input('Punto inicial (x0)', value=1.0, format="%.4f")
                delta = st.number_input('Incremento (Δ)', value=0.1, format="%.4f")
            if st.button("Ejecutar", type="primary", use_container_width=True):
                f = info_funcion['intervalos'][0][2]
                intervalo_a, intervalo_b, fig = bounding_phase_method(f, x0, delta)
                with results_container:
                    if intervalo_a is not None:
                        st.success(f"Intervalo acotado encontrado: [{intervalo_a:.6f}, {intervalo_b:.6f}]")
                        st.pyplot(fig)
                    else:
                        st.warning("No se pudo acotar un mínimo. Prueba con otros parámetros.")
        
        elif algoritmo_seleccionado == 'Región de Eliminaciones':
            with st.expander("Parámetros del Algoritmo"):
                tolerancia = st.number_input('Tolerancia', value=0.01, format="%.6f", min_value=0.000001)
            if st.button("Ejecutar", type="primary", use_container_width=True):
                a, b, f = info_funcion['intervalos'][0]
                intervalo_a, intervalo_b, fig = region_elimination(f, a, b, precision=tolerancia)
                with results_container:
                    st.success(f"Intervalo final: [{intervalo_a:.6f}, {intervalo_b:.6f}]")
                    st.markdown(f"**Longitud del intervalo:** `{intervalo_b - intervalo_a:.6f}`")
                    st.pyplot(fig)

        elif algoritmo_seleccionado == 'Intervalos a la Mitad':
            with st.expander("Parámetros del Algoritmo"):
                epsilon = st.number_input('Precisión (ε)', value=0.01, format="%.6f", help="Longitud deseada del intervalo final.")
            if st.button("Ejecutar", type="primary", use_container_width=True):
                a, b, f = info_funcion['intervalos'][0]
                ia, ib, fig = intervalos_a_la_mitad(f, a, b, epsilon=epsilon)
                with results_container:
                    st.success(f"Intervalo final acotado: $[{ia:.6f}, {ib:.6f}]$")
                    st.pyplot(fig)

        elif algoritmo_seleccionado == 'Fibonacci Search':
            with st.expander("Parámetros del Algoritmo"):
                n = st.number_input('Número de evaluaciones (n)', min_value=3, value=10, step=1)
            if st.button("Ejecutar", type="primary", use_container_width=True):
                a, b, f = info_funcion['intervalos'][0]
                ia, ib, fig = fibonacci_search(f, a, b, n_evaluations=n)
                with results_container:
                    st.success(f"Intervalo final acotado: $[{ia:.6f}, {ib:.6f}]$")
                    st.pyplot(fig)
        
        elif algoritmo_seleccionado == 'Golden Section Search':
            with st.expander("Parámetros del Algoritmo"):
                epsilon = st.number_input('Precisión (ε)', value=0.01, format="%.6f")
            if st.button("Ejecutar", type="primary", use_container_width=True):
                a, b, f = info_funcion['intervalos'][0]
                ia, ib, fig = golden_section_search(f, a, b, epsilon=epsilon)
                with results_container:
                    st.success(f"Intervalo final acotado: $[{ia:.6f}, {ib:.6f}]$")
                    st.pyplot(fig)
        
        elif algoritmo_seleccionado == 'Newton-Raphson':
            if 'df' not in info_funcion or 'ddf' not in info_funcion:
                st.warning("Esta función no tiene las derivadas necesarias definidas.")
            else:
                with st.expander("Parámetros del Algoritmo"):
                    x0 = st.number_input('Suposición inicial (x0)', value=1.0)
                    epsilon = st.number_input('Tolerancia (ε)', value=0.0001, format="%.6f")
                if st.button("Ejecutar", type="primary", use_container_width=True):
                    f, df, ddf = info_funcion['intervalos'][0][2], info_funcion['df'], info_funcion['ddf']
                    xp, fp, fig = newton_raphson(f, df, ddf, x0, epsilon=epsilon)
                    with results_container:
                        if xp is not None:
                            st.success(f"Punto mínimo encontrado en x ≈ {xp:.6f} con f(x) ≈ {fp:.6f}")
                            st.pyplot(fig)
                        else:
                            st.error("El algoritmo no convergió.")

        elif algoritmo_seleccionado == 'Método de Bisección':
            if 'df' not in info_funcion:
                st.warning("Este método requiere la primera derivada de la función, definida como 'df'.")
            else:
                with st.expander("Parámetros del Algoritmo"):
                    a, b, _ = info_funcion['intervalos'][0]
                    a_biseccion = st.number_input('Límite inferior (a)', value=a)
                    b_biseccion = st.number_input('Límite superior (b)', value=b)
                    epsilon = st.number_input('Tolerancia (ε)', value=0.0001, format="%.6f")
                if st.button("Ejecutar", type="primary", use_container_width=True):
                    f, df = info_funcion['intervalos'][0][2], info_funcion['df']
                    xp, fp, fig = metodo_biseccion(f, df, a_biseccion, b_biseccion, epsilon)
                    with results_container:
                        if xp is not None:
                            st.success(f"Punto crítico aproximado: x ≈ {xp:.6f} con f(x) ≈ {fp:.6f}")
                            st.pyplot(fig)
                        else:
                            st.error("No se pudo converger. Asegúrate de que f'(a) y f'(b) tengan signos opuestos.")

        elif algoritmo_seleccionado == 'Método Secante':
            if 'df' not in info_funcion:
                st.warning("Este método requiere la primera derivada de la función, definida como 'df'.")
            else:
                with st.expander("Parámetros del Algoritmo"):
                    a, b, _ = info_funcion['intervalos'][0]
                    x1 = st.number_input('Primer punto inicial (x1)', value=a)
                    x2 = st.number_input('Segundo punto inicial (x2)', value=b)
                    epsilon = st.number_input('Tolerancia (ε)', value=0.0001, format="%.6f")
                if st.button("Ejecutar", type="primary", use_container_width=True):
                    f, df = info_funcion['intervalos'][0][2], info_funcion['df']
                    xp, fp, fig = metodo_secante(f, df, x1, x2, epsilon)
                    with results_container:
                        if xp is not None:
                            st.success(f"Punto crítico aproximado: x ≈ {xp:.6f} con f(x) ≈ {fp:.6f}")
                            st.pyplot(fig)
                        else:
                            st.error("El algoritmo no convergió.")

        elif algoritmo_seleccionado == 'Búsqueda Unidireccional':
            with st.expander("Parámetros del Algoritmo"):
                st.markdown("##### Punto Inicial")
                c1, c2 = st.columns(2)
                x_t_x = c1.number_input('x_t (coordenada x)', value=1.0)
                x_t_y = c2.number_input('x_t (coordenada y)', value=1.0)
                st.markdown("##### Dirección de Búsqueda")
                c3, c4 = st.columns(2)
                s_t_x = c3.number_input('s_t (componente x)', value=1.0)
                s_t_y = c4.number_input('s_t (componente y)', value=1.0)
                st.markdown("##### Rango de Búsqueda (α)")
                c5, c6 = st.columns(2)
                a_min = c5.number_input('α mínimo', value=-2.0)
                a_max = c6.number_input('α máximo', value=2.0)
                n_points = st.slider('Número de puntos para búsqueda lineal', 10, 1000, 100)
            if st.button("Ejecutar", type="primary", use_container_width=True):
                f_multi = info_funcion['funcion']
                x_t = np.array([x_t_x, x_t_y])
                s_t = np.array([s_t_x, s_t_y])
                x_opt, fig = unidireccional_search(f_multi, x_t, s_t, exhaustive_search, (a_min, a_max), n_points)
                with results_container:
                    st.success(f"Punto óptimo: ({x_opt[0]:.6f}, {x_opt[1]:.6f}), f(x) ≈ {f_multi(x_opt):.6f}")
                    st.pyplot(fig)

        elif algoritmo_seleccionado == 'Caminata Aleatoria':
            with st.expander("Parámetros del Algoritmo"):
                c1, c2 = st.columns(2)
                x0_x = c1.number_input('Punto inicial x', value=1.0)
                x0_y = c2.number_input('Punto inicial y', value=1.0)
                epsilon = c1.number_input('Tolerancia (ε)', value=0.01, format="%.6f")
                step_size = c2.number_input('Tamaño de paso (step)', value=0.5)
                max_iter = st.number_input('Máx. iteraciones', value=1000, step=100)
            if st.button("Ejecutar", type="primary", use_container_width=True):
                f_multi = info_funcion['funcion']
                x0 = np.array([x0_x, x0_y])
                x_best, f_best, fig = caminata_aleatoria(f_multi, x0, epsilon=epsilon, max_iter=max_iter, step_size=step_size)
                with results_container:
                    st.success(f"Mejor punto: ({x_best[0]:.6f}, {x_best[1]:.6f}), f(x) ≈ {f_best:.6f}")
                    st.pyplot(fig)

        elif algoritmo_seleccionado == 'Hill Climbing':
            with st.expander("Parámetros del Algoritmo"):
                c1, c2 = st.columns(2)
                x0_x = c1.number_input('Punto inicial x', value=1.0)
                x0_y = c2.number_input('Punto inicial y', value=1.0)
                epsilon = c1.number_input('Tolerancia (ε)', value=0.01, format="%.6f")
                step_size = c2.number_input('Tamaño de paso', value=0.5)
                max_iter = st.number_input('Máx. iteraciones', value=1000, step=100)
            if st.button("Ejecutar", type="primary", use_container_width=True):
                f_multi = info_funcion['funcion']
                x0 = np.array([x0_x, x0_y])
                x_best, f_best, fig = hill_climbing(f_multi, x0, epsilon=epsilon, max_iter=max_iter, step_size=step_size)
                with results_container:
                    st.success(f"Mejor punto: ({x_best[0]:.6f}, {x_best[1]:.6f}), f(x) ≈ {f_best:.6f}")
                    st.pyplot(fig)

        elif algoritmo_seleccionado == 'Recocido Simulado':
            with st.expander("Parámetros del Algoritmo"):
                c1, c2 = st.columns(2)
                x0_x = c1.number_input('Punto inicial x', value=1.0)
                T0 = c1.number_input('Temperatura inicial (T0)', value=100.0)
                w_size = c1.number_input('Tamaño de metrópolis (w)', value=100)
                x0_y = c2.number_input('Punto inicial y', value=1.0)
                alpha = c2.number_input('Factor de enfriamiento (α)', value=0.95, help="Debe ser < 1")
                max_iter = c2.number_input('Máx. iteraciones', value=1000)
            if st.button("Ejecutar", type="primary", use_container_width=True):
                f_multi = info_funcion['funcion']
                x0 = np.array([x0_x, x0_y])
                x_best, f_best, fig = recocido_simulado(f_multi, x0, T0=T0, alpha=alpha, w_size=w_size, max_iter=max_iter)
                with results_container:
                    st.success(f"Mejor punto: ({x_best[0]:.6f}, {x_best[1]:.6f}), f(x) ≈ {f_best:.6f}")
                    st.pyplot(fig)

        elif algoritmo_seleccionado == 'Nelder-Mead':
            with st.expander("Parámetros del Algoritmo"):
                c1, c2 = st.columns(2)
                x0_x = c1.number_input('Punto inicial x', value=1.0)
                delta = c1.number_input('Tamaño del simplex (δ)', value=1.0)
                beta = c1.number_input('Contracción (β)', value=0.5, min_value=0.0, max_value=1.0)
                max_iter = c1.number_input('Máx. iteraciones', value=200, min_value=10, step=10)
                x0_y = c2.number_input('Punto inicial y', value=1.0)
                gamma = c2.number_input('Expansión (γ)', value=2.0, min_value=1.0)
                epsilon = c2.number_input('Tolerancia (ε)', value=0.0001, format="%.6f")
            if st.button("Ejecutar", type="primary", use_container_width=True):
                f, intervalos = info_funcion['funcion'], info_funcion['intervalos']
                x0 = np.array([x0_x, x0_y])
                x_min, f_min, fig = nelder_mead(f, x0, intervalos, delta=delta, gamma=gamma, beta=beta, epsilon=epsilon, max_iter=max_iter)
                with results_container:
                    st.success(f"Mínimo encontrado: ({x_min[0]:.6f}, {x_min[1]:.6f}), f(x) ≈ {f_min:.6f}")
                    st.pyplot(fig)
        
        elif algoritmo_seleccionado == 'Método de Hooke-Jeeves':
            with st.expander("Parámetros del Algoritmo"):
                c1, c2 = st.columns(2)
                x0_x = c1.number_input('Punto inicial x', value=0.0)
                delta = c1.number_input('Tamaño de paso (δ)', value=1.0)
                epsilon = c1.number_input('Tolerancia (ε)', value=0.0001, format="%.6f")
                x0_y = c2.number_input('Punto inicial y', value=0.0)
                alpha = c2.number_input('Factor de reducción (α > 1)', value=2.0)
                max_iter = st.number_input('Máx. iteraciones', value=200)
            if st.button("Ejecutar", type="primary", use_container_width=True):
                f, intervalos = info_funcion['funcion'], info_funcion['intervalos']
                x0 = np.array([x0_x, x0_y])
                x_min, f_min, fig = hooke_jeeves(f, x0, intervalos, deltas=delta, alpha=alpha, epsilon=epsilon, max_iter=max_iter)
                with results_container:
                    st.success(f"Mínimo encontrado: ({x_min[0]:.6f}, {x_min[1]:.6f}), f(x) ≈ {f_min:.6f}")
                    st.pyplot(fig)

        elif algoritmo_seleccionado == 'Método del Gradiente':
            st.info("El Método del Gradiente está como placeholder.")
            st.code("""
def metodo_gradiente(funcion, gradiente, x0, intervalos):
    # Lógica a implementar aquí
    pass
            """, language='python')
            if st.button("Ejecutar", type="primary", use_container_width=True):
                with results_container:
                    st.success("Ejecución simulada con éxito.")

        elif algoritmo_seleccionado == 'Método de Cauchy':
            with st.expander("Parámetros del Algoritmo"):
                c1, c2 = st.columns(2)
                x0_x = c1.number_input('Punto inicial x', value=0.0)
                epsilon1 = c1.number_input('Tolerancia ε1', value=0.0001, format="%.6f")
                max_iter = c1.number_input('Máx. iteraciones', value=100)
                x0_y = c2.number_input('Punto inicial y', value=0.0)
                epsilon2 = c2.number_input('Tolerancia ε2', value=0.0001, format="%.6f")
                max_alpha = c2.number_input('α máximo', value=2.0)
            if st.button("Ejecutar", type="primary", use_container_width=True):
                f, grad, intervalos = info_funcion['funcion'], info_funcion['gradiente'], info_funcion['intervalos']
                x0 = np.array([x0_x, x0_y])
                x_min, f_min, fig = metodo_cauchy(f, grad, x0, intervalos, epsilon1, epsilon2, max_iter, max_alpha)
                with results_container:
                    st.success(f"Mínimo encontrado: ({x_min[0]:.6f}, {x_min[1]:.6f}), f(x) ≈ {f_min:.6f}")
                    st.pyplot(fig)

        elif algoritmo_seleccionado == 'Método de Newton':
            with st.expander("Parámetros del Algoritmo"):
                c1, c2 = st.columns(2)
                x0_x = c1.number_input('Punto inicial x', value=0.0)
                epsilon1 = c1.number_input('Tolerancia ε1', value=0.0001, format="%.6f")
                delta_hess = c1.number_input('Delta Hessiana', value=1e-5, format="%.6f")
                x0_y = c2.number_input('Punto inicial y', value=0.0)
                epsilon2 = c2.number_input('Tolerancia ε2', value=0.0001, format="%.6f")
                max_alpha = c2.number_input('α máximo', value=2.0)
                max_iter = st.number_input('Máx. iteraciones', value=100)
            if st.button("Ejecutar", type="primary", use_container_width=True):
                f, grad, intervalos = info_funcion['funcion'], info_funcion['gradiente'], info_funcion['intervalos']
                x0 = np.array([x0_x, x0_y])
                x_min, f_min, fig = metodo_newton(f, grad, x0, intervalos, epsilon1, epsilon2, delta_hess, max_iter, max_alpha)
                with results_container:
                    st.success(f"Mínimo encontrado: ({x_min[0]:.6f}, {x_min[1]:.6f}), f(x) ≈ {f_min:.6f}")
                    st.pyplot(fig)

    with tab2:
        st.header(f"Detalles de la Función: {funcion_seleccionada}", divider="orange")
        st.markdown(f"**Expresión Matemática:**")
        st.latex(info_funcion['latex'])
        
        if es_multivariado:
            st.markdown(f"**Dominio:**")
            st.code(info_funcion['dominio'], language="latex")
            st.markdown(f"**Mínimo Global Conocido:**")
            st.code(info_funcion['minimo'], language="latex")
            st.markdown(f"**Coordenadas del Mínimo:**")
            st.code(str(info_funcion['minimo_coords']))

        else: # Univariada
            if "latex_derivada" in info_funcion:
                st.markdown(f"**Derivada:**")
                st.latex(info_funcion['latex_derivada'])
            st.markdown(f"**Dominio/Intervalo Inicial:**")
            st.code(info_funcion['dominio'], language="latex")
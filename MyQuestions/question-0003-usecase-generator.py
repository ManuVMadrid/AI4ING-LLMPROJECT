import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import make_regression

def generar_caso_de_uso_evaluar_eficiencia_energetica():
    """
    Genera un caso de uso aleatorio para la función evaluar_eficiencia_energetica.
    Retorna una tupla: (diccionario_de_argumentos, output_esperado)
    """
    
    # ---------------------------------------------------------
    # 1. Configuración Aleatoria de las Entradas (Input)
    # ---------------------------------------------------------
    
    # Generamos de manera procedural (con make_regression) un dataset aleatorio 
    # simulando medidas estructurales (X) y carga térmica (y).
    # Variamos el tamaño de la muestra ligeramente cada vez.
    n_muestras = np.random.randint(150, 300)
    
    X, y = make_regression(
        n_samples=n_muestras, 
        n_features=5,       # Simulamos 5 variables constructivas
        n_informative=4,    # 4 son útiles, 1 es puro ruido
        noise=10.0,         # Le añadimos ruido estadístico para que no sea un problema perfecto
        random_state=None   # None para que sea aleatorio en cada corrida
    )
    
    # ---------------------------------------------------------
    # 2. Construir el diccionario de INPUT
    # ---------------------------------------------------------
    # TIP IMPORTANTE: Usamos .copy() para preservar la pureza de los datos originales.
    
    input_dict = {
        'X': X.copy(),
        'y': y.copy()
    }
    
    # ---------------------------------------------------------
    # 3. Lógica del GROUND TRUTH (Resultado Esperado)
    # ---------------------------------------------------------
    
    # A. Dividimos los datos asegurando el mismo random_state exigido en las instrucciones (42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # B. Instanciamos y aplicamos el StandardScaler
    scaler = StandardScaler()
    # Importante: El fit (ajuste de la media y varianza teórica) SOLAMENTE se hace con X_train para evitar data leakage
    X_train_scaled = scaler.fit_transform(X_train)
    # El test se transforma con los parámetros ya aprendidos
    X_test_scaled = scaler.transform(X_test)
    
    # C. Entrenamos el modelo KNeighborsRegressor
    modelo_knn = KNeighborsRegressor(n_neighbors=5)
    modelo_knn.fit(X_train_scaled, y_train)
    
    # D. Calculamos predicciones y evaluamos con el Error Absoluto Medio
    predicciones = modelo_knn.predict(X_test_scaled)
    mae_resultado = mean_absolute_error(y_test, predicciones)
    
    # Retornamos el número exacto del MAE
    expected_output = float(mae_resultado)
    
    return input_dict, expected_output


# --- Bloque principal para comprobar que el código funciona localmente ---
if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_evaluar_eficiencia_energetica()
    
    print("=== DICCIONARIO INPUT ===")
    print(f"Forma de X (Características constructivas): {entrada['X'].shape}")
    print(f"Forma de y (Carga térmica requerida): {entrada['y'].shape}")
    print("Primeros 2 registros de X:")
    print(entrada['X'][:2])
    
    print("\n=== OUTPUT ESPERADO (Ground Truth MAE) ===")
    print(salida)

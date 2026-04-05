import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def generar_caso_de_uso_clasificar_spam_metadata():
    """
    Genera un caso de uso aleatorio para la función clasificar_spam_metadata.
    Retorna una tupla: (diccionario_de_argumentos, output_esperado)
    """
    
    # ---------------------------------------------------------
    # 1. Configuración Aleatoria de las Entradas (Input)
    # ---------------------------------------------------------
    
    # Construimos un dataset sintético de clasificación binaria (Spam / No Spam)
    n_muestras = np.random.randint(150, 400)
    
    X, y = make_classification(
        n_samples=n_muestras,
        n_features=6,        # Simulando metadatos: contador enlaces, contador mayúsculas, etc.
        n_informative=4,     # 4 características informativas
        n_classes=2,         # 2 Clases: 0 (No Spam) y 1 (Spam)
        random_state=None    # Dinamismo
    )
    
    # Como la rúbrica exige que los datos ya le lleguen partidos al estudiante, los partimos aquí mismo.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # ---------------------------------------------------------
    # 2. Construir el diccionario de INPUT
    # ---------------------------------------------------------
    
    # Copiamos para no mutar el estado global de la prueba
    input_dict = {
        'X_train': X_train.copy(),
        'y_train': y_train.copy(),
        'X_test': X_test.copy(),
        'y_test': y_test.copy()
    }
    
    # ---------------------------------------------------------
    # 3. Lógica del GROUND TRUTH (Resultado Esperado)
    # ---------------------------------------------------------
    
    # A. Escalamos con MinMaxScaler para poner todo entre 0 y 1
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # B. Entrenamos el Gaussian Naive Bayes (No usa argumentos explícitos usualmente)
    modelo_nb = GaussianNB()
    modelo_nb.fit(X_train_scaled, y_train)
    
    # C. Realizamos la predicción 
    predicciones = modelo_nb.predict(X_test_scaled)
    
    # D. Devolvemos el puntaje de exactitud usando metrics
    accuracy = accuracy_score(y_test, predicciones)
    expected_output = float(accuracy)
    
    return input_dict, expected_output


# --- Bloque principal para comprobar que el código funciona localmente ---
if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_clasificar_spam_metadata()
    
    print("=== DICCIONARIO INPUT ===")
    print(f"Dimensiones de X_train (Metadatos de Entrenamiento): {entrada['X_train'].shape}")
    print(f"Dimensiones de X_test (Metadatos de Prueba): {entrada['X_test'].shape}")
    
    print("\n=== OUTPUT ESPERADO (Accuracy / Exactitud) ===")
    # Veremos el resultado porcentual del modelo sobre el grupo test
    print(salida)

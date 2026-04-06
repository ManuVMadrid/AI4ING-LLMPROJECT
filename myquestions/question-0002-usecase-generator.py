import pandas as pd
import numpy as np
import random
import string

def generar_caso_de_uso_evaluar_becas_estudiantes():
    """
    Genera un caso de uso aleatorio para la función evaluar_becas_estudiantes.
    Retorna una tupla: (diccionario_de_argumentos, output_esperado)
    """
    
    # ---------------------------------------------------------
    # 1. Configuración Aleatoria de las Entradas (Input)
    # ---------------------------------------------------------
    
    n_estudiantes = random.randint(15, 30)
    datos_notas = []
    
    # Simular una cantidad n de estudiantes con notas aleatorias
    for i in range(1, n_estudiantes + 1):
        # Generar un ID estudiante estilo "EST-A8X"
        sufijo = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
        id_estudiante = f"EST-{sufijo}"
        
        # Generar notas entre 40 y 100
        nota_m = random.randint(40, 100)
        nota_c = random.randint(40, 100)
        nota_l = random.randint(40, 100)
        
        datos_notas.append({
            'id_estudiante': id_estudiante,
            'matematicas': float(nota_m),
            'ciencias': float(nota_c),
            'literatura': float(nota_l)
        })
        
    df_notas = pd.DataFrame(datos_notas)
    
    # Introducimos NaNs (valores nulos simulando inasistencias) aleatorios 
    # en aproximadamente un 15% de los datos numéricos
    mask = np.random.choice([True, False], size=(df_notas.shape[0], 3), p=[0.15, 0.85])
    df_notas.loc[:, ['matematicas', 'ciencias', 'literatura']] = df_notas[['matematicas', 'ciencias', 'literatura']].mask(mask)
    
    # Para garantizar una distribución no ordenada, desordenamos el DataFrame artificialmente
    df_notas = df_notas.sample(frac=1).reset_index(drop=True)
    
    # ---------------------------------------------------------
    # 2. Construir el diccionario de INPUT
    # ---------------------------------------------------------
    # TIP IMPORTANTE: Usamos .copy() para preservar la pureza de los datos
    input_dict = {
        'df_notas': df_notas.copy()
    }
    
    # ---------------------------------------------------------
    # 3. Lógica del GROUND TRUTH (Resultado Esperado)
    # ---------------------------------------------------------
    
    # Trabajamos sobre una copia temporal para construir el objetivo
    df_resultado = df_notas.copy()
    
    # Reemplazamos todos los NaN por el valor 0 (como indicó la rúbrica)
    df_resultado.fillna(0, inplace=True)
    
    # Aplicamos sumatoria combinada ponderada
    df_resultado['nota_final'] = (
        df_resultado['matematicas'] * 0.40 + 
        df_resultado['ciencias'] * 0.40 + 
        df_resultado['literatura'] * 0.20
    )
    
    # Filtramos por aquellos con una nota >= 80 y redondeamos
    df_filtrado = df_resultado[df_resultado['nota_final'] >= 80].copy()
    df_filtrado['nota_final'] = df_filtrado['nota_final'].round(2)
    
    # Retornamos solo las dos columnas indicadas, y nos aseguramos de 
    # ordenarlo alfabéticamente por 'id_estudiante'
    expected_output = df_filtrado[['id_estudiante', 'nota_final']].sort_values(by='id_estudiante').reset_index(drop=True)
    
    return input_dict, expected_output


# --- Bloque principal para comprobar que el código funciona localmente ---
if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_evaluar_becas_estudiantes()
    
    print("=== DICCIONARIO INPUT ===")
    print("DataFrame de Notas generadas (mostrar posibles NaNs):")
    print(entrada['df_notas'].head(8))
    
    print("\n=== OUTPUT ESPERADO (Ground Truth - Becados) ===")
    print(salida)

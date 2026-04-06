import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generar_caso_de_uso_resumir_ganancias_helados():
    """
    Genera un caso de uso aleatorio para la función resumir_ganancias_helados.
    Retorna una tupla: (diccionario_de_argumentos, output_esperado)
    """
    
    # ---------------------------------------------------------
    # 1. Configuración Aleatoria de las Entradas (Input)
    # ---------------------------------------------------------
    
    sabores_disponibles = ['Vainilla', 'Chocolate', 'Fresa', 'Limon', 'Menta', 'Pistacho']
    n_sabores = random.randint(3, 6)
    sabores = random.sample(sabores_disponibles, n_sabores)
    
    # A. Generar df_costos aleatorio
    datos_costos = []
    for sabor in sabores:
        costo = round(random.uniform(1.0, 3.0), 2)  # Costo entre 1 y 3 dólares
        precio = round(costo + random.uniform(1.5, 4.0), 2)  # Precio con buen margen
        datos_costos.append({'sabor': sabor, 'costo_produccion': costo, 'precio_venta': precio})
        
    df_costos = pd.DataFrame(datos_costos)
    
    # B. Generar df_ventas aleatorio (simulamos 15 a 30 ventas)
    n_ventas = random.randint(15, 30)
    datos_ventas = []
    fecha_inicial = datetime(2026, 1, 1)
    
    for _ in range(n_ventas):
        # Fecha aleatoria
        fecha_str = (fecha_inicial + timedelta(days=random.randint(0, 30))).strftime('%Y-%m-%d')
        # Sabor aleatorio que sí exista en los costos
        sabor = random.choice(sabores)
        # Temperatura entre 15 y 35 (algunos pasarán el umbral de 25, otros no)
        temperatura = round(random.uniform(15.0, 35.0), 1)
        # Unidades vendidas
        unidades = random.randint(10, 100)
        
        datos_ventas.append({
            'fecha': fecha_str, 
            'sabor': sabor, 
            'temperatura': temperatura, 
            'unidades_vendidas': unidades
        })
        
    df_ventas = pd.DataFrame(datos_ventas)
    
    # ---------------------------------------------------------
    # 2. Construir el diccionario de INPUT
    # ---------------------------------------------------------
    # TIP IMPORTANTE: Usamos .copy() para que la función evaluadora 
    # no modifique nuestros datos originales generados.
    
    input_dict = {
        'df_ventas': df_ventas.copy(),
        'df_costos': df_costos.copy()
    }
    
    # ---------------------------------------------------------
    # 3. Lógica del GROUND TRUTH (Resultado Esperado)
    # ---------------------------------------------------------
    
    # Replicamos el código que se espera que el compañero escriba:
    df_merge = pd.merge(df_ventas, df_costos, on='sabor')
    
    # Filtramos días con más de 25 grados
    df_calor = df_merge[df_merge['temperatura'] > 25].copy()
    
    if df_calor.empty:
        # En caso estadísticamente raro de que ningún día supere los 25 grados
        expected_output = pd.Series(dtype=float, name='ganancia_neta')
        expected_output.index.name = 'sabor'
    else:
        # Calculamos matemática básica: unidades * (precio - costo)
        df_calor['ganancia_neta'] = df_calor['unidades_vendidas'] * (df_calor['precio_venta'] - df_calor['costo_produccion'])
        
        # Agrupamos por sabor y sumamos la retribución
        expected_output = df_calor.groupby('sabor')['ganancia_neta'].sum().sort_values(ascending=False)
    
    return input_dict, expected_output


# --- Bloque principal para comprobar que el código funciona localmente ---
if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_resumir_ganancias_helados()
    
    print("=== DICCIONARIO INPUT ===")
    print("1. DataFrame de Ventas (primeras 5 filas):")
    print(entrada['df_ventas'].head())
    print("\n2. DataFrame de Costos:")
    print(entrada['df_costos'])
    
    print("\n=== OUTPUT ESPERADO (Ground Truth) ===")
    print(salida)

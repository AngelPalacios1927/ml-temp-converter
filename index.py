import tensorflow as tf 
import numpy as np 

# Datos de entrenamiento
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# Definicion del modelo
capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

# Compilacion del modelo
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

# Entrenamiento del modelo
print("Iniciando entrenamiento...")
modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print("Entrenamiento finalizado.")

# Bucle para realizar predicciones
while True:
    # Solicitar al usuario que ingrese un valor de Celsius
    entrada = input("Introduce un valor en grados Celsius (o escribe 'salir' para terminar): ")
    
    if entrada.lower() == 'salir':
        break
    
    try:
        celsius_input = float(entrada)
        resultado = modelo.predict(np.array([celsius_input]))
        print(f"{celsius_input} grados Celsius son aproximadamente {resultado[0][0]} grados Fahrenheit")
    except ValueError:
        print("Por favor, introduce un número válido.")

print("Programa finalizado.")

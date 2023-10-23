import tensorflow as tf
from tensorflow import keras
import numpy as np

# Elemento 1: Modelado de Incertidumbre
# Suponemos datos con cierta incertidumbre.
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)  # Modelo lineal con ruido.

# Elemento 4: Aprendizaje Automático Probabilístico
# Creamos una red neuronal con una capa oculta.
model = keras.Sequential([
    keras.layers.Dense(units=10, activation='relu', input_shape=(1,)),
    keras.layers.Dense(units=1)  # Capa de salida para regresión.
])

# Elemento 5: Toma de Decisiones Probabilística
# Compilamos el modelo especificando una función de pérdida y optimizador.
model.compile(loss='mean_squared_error', optimizer='adam')

# Elemento 6: Robustez y Adaptabilidad
# Elemento 7: Control y Toma de Decisiones

# Elemento 8: Redes Neuronales Profundas
# Entrenamos el modelo
model.fit(X, y, epochs=100, verbose=0)

# Elemento 1: Modelado de Incertidumbre

# Elemento 2: Redes Bayesianas (Opcional)
# Elemento 3: Inferencia Probabilística (Opcional)

# Evaluamos el modelo en nuevos datos
X_test = np.array([[0.5], [0.7], [0.9]])
y_pred = model.predict(X_test)

# Elemento 4: Aprendizaje Automático Probabilístico

# Elemento 5: Toma de Decisiones Probabilística

# Elemento 6: Robustez y Adaptabilidad
# Elemento 7: Control y Toma de Decisiones

# Elemento 8: Redes Neuronales Profundas (Opcional)

# Imprimimos las predicciones
print("Predicciones:")
for i, x in enumerate(X_test):
    print(f"Entrada: {x}, Predicción: {y_pred[i][0]}")

# Visualización opcional (requiere la biblioteca matplotlib)
import matplotlib.pyplot as plt
plt.scatter(X, y, label='Datos reales')
plt.plot(X_test, y_pred, color='red', label='Predicciones')
plt.legend()
plt.xlabel('X')
plt.ylabel('y')
plt.title('Predicciones de Regresión con Red Neuronal')
plt.show()

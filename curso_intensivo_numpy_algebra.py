print("******")
print("NUMPY")
print("******")
print("\n")

# Paquete de compuación eficiente (escrito en C) enfocado a álgebra lineal
# Basado en vectores (arrays de una sola dimension) y matrices (de dos diemensiones)

import numpy as np
import random as rd

# Python no tiene ninguna estructura para trabajar con datos de manera masiva
# Orientada a almacenar y recuperar información, pero no a procesarla de manera eficiente
# Los modelos de ML e análisis se basan en el concepto vector (lo más parecido una lista en Pyhon) pero no tiene operaciones vectorizadas (parecido a la función map())

# Si multiplicamos una lista en Python:

lista = [1,2,3]
print(lista * 2)

# Para multiplica cadaa elemento de la lista deberíamos utilizar una función lambda, p.e.

print([elemento * 2 for elemento in lista])

# Sin embargo, si trabajamos con un array de numpy:

vector = np.array([1,2,3])
print(vector * 2)

# A nivel de estructura de datos numpy utiliza el array multidimensional (array de n dimensiones)

print(type(vector))
print(vector.dtype) # Es importante conocer cómo define el dato (por ejemplo, int) más el número de bits (16, 32, 64, etc.)

# Aunque numpy sea una librería numérica también puede haber arrays no numéricos
# Pero todos los componentes de un array tienen que ser del mismo tipo (en Pandas si es posible)

vector = np.array(["A", "B", "C"])
print(type(vector)) 

vector = np.array(["A", 2, "C"])
print(vector) # El 2 lo convierte a cadena de texto

print("\n")
print("*******************")
print("COMO CREAR UN ARRAY")
print("*******************")
print("\n")

vector = np.array([1,2,3])
print(vector)

vector = np.array([[1,2,3],[4,5,6]]) # Una matriz
print(vector)

# Crear un vecto a partir de un rango
vector = np.arange(1,10,2) 
print(vector)

# Crear un vector interpolado entre dos números (20 es el número de elementos)
vector = np.linspace(1,10,20) 
print(vector)

# Crear un array de ceros (el parámetro es una tupla con filas-columnas)
vector = np.zeros((4,3)) 
print(vector)

# Crear una matriz de idenidad (matriz cuadrada de 0 y 1 en coincidentes fila-columna)
vector = np.eye(5) 
print(vector)

# Como crear un array aleatorio. Si lo hacemos utilizando el módulo rand de Python tenemos el mismo problema
# No está preparado para operar con este tipo de estructuras de datos y aunque se puede encontrar soluciones como:

vector = rd.randint(1,10) # Devuelve un solo valor
print(vector)

# Si quisiéramos genera 10 valores
print([rd.randint(1,10) for cada in range(10)])

# Pero en numpy:

# Aleatorios de 0-1 con distribución uniforme.
# Una distribución mide cual es la frecuencia de aparición de cada nombre

vector = np.random.rand(5,10) # Con el método random de numpy
print(vector)

# Una distribución normal con media 0
vector = np.random.randn(20)
print(vector)

# Una distribución uniforme (campana de Gauss) entre dos int
vector = np.random.randint(1,11,20)
print(vector)

# O también es muy útil establecer una semilla
# Es decir, que la simulación de números aleatorios (pseudoaleatorio) sean los mismos partiendo de la misma semilla
np.random.seed(1234)
vector = np.random.randint(1,11,20)
print(vector)

# Para ver la forma de un array
vector = np.array([1,2,3])

# Acceder a la propiedad de un array, en este caso array multidimensional
# Si saliera solo el 3 sería un escalar, de ahí la , que indica que es multidimensional, pero de 1 dimension

print(vector.shape) 

# Para cambiar la forma de un array
vector = np.arange(20)
print(vector)
print(vector.shape) 

vector2 = vector.reshape(2,10) # Ahora es un array de 2 filas y 10 columnas
print(vector2.shape) 

# Para ordenar un array
vector = np.random.rand(10)
print(vector)

vector.sort()
print(vector)

print("\n")
print("********************")
print("ESTADÍSTICOS BÁSICOS")
print("********************")
print("\n")

vector = np.arange(20)
print(vector)

print(vector.mean()) # La media
print(np.median(vector)) # La mediana
print(np.std(vector)) # Desviación típica
print(vector.var()) # Varianza
print(vector.max()) # Máximo
print(vector.min()) # Mínimo
print(np.corrcoef(vector,vector)) # Matriz de correlación

# Para localizar los índices de los estadísticos (sobre todo para los maximos y los mínimos)
# Por ejemplo de este array de 20 números localizar en que posición está el máximo:

vector = np.random.randint(1,11,20)
print(vector) # Solo nos da el primer máximo que se encuentra, aunqu haya dos
print(vector.argmax()) # O argmin

# Y luego utilizar esa posición para recuperar el valor
print(vector[vector.argmax()])

print("\n")
print("*********************")
print("CREAR COPIAS DE DATOS")
print("*********************")
print("\n")

# Esta parte es importante para cuando queremos aplicar operaciones a un dataset sin modificar el original

vector1 = np.arange(5)
print(vector1)

# Con esta operación pensaríamos que estamos guardamos todo el contenido de vector1 en vector2
# Lo que en realidad estamos haciendo es crear un puntero vector2 al contenido de vector1

vector2 = vector1 

# Por eso cuando modificamos el contenido en vector también modificamos el contenido en vector1

vector2[:] = 5 # Que todos los elementos sean 5

print(vector2)
print(vector1)

# La manera correcta de hacerlo sería:
vector1 = np.arange(5)
print(vector1)

vector2 = vector1.copy() # Para hacer una copia de los datos
vector2[:] = 5 # Que todos los elementos sean 5

print(vector2)
print(vector1) # Ahora vector1 mantiene el contenido original

print("\n")
print("**************")
print("INDEXAR ARRAYS")
print("**************")
print("\n")

vector = np.arange(11)
print(vector[2])
print(vector[2:5])

# En el caso de la matrices:

matriz = np.random.rand(3,4)
print(matriz)

# Para recuperar un dato en concreto

# OPCIÓN 1. Con dos corchetes, el primero para las filas y el segundo para las columnas
print(matriz[0][0])

# OPCIÓN 2. Con un solo corchete, pero filas y columnas separadas por comas
print(matriz[0,0])

# Para recuperar un rango completo, por ejemplo una fila o una columna

print(matriz[0,:])
print(matriz[:,0])

# Para recuperar un rango parcial de una fila o una columna

print(matriz[0,0:2])
print(matriz[0:1,0])

# En numpy también podemos indexar por vectores bool
# Como las condiciones generan vectores bool, entonces:

vector = np.random.rand(10)
print(vector)
print(vector < 0.5) # Me muestra un vector con el resultado de la comparación

# Y podemos indexar por esa condición, p.e. cuando queremos comprobar un criterio
vector = np.random.rand(10)
print(vector)
print(vector[vector < 0.5]) # Me muestra un vector con el resultado de la comparación







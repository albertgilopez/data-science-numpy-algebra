print("**************************")
print("EJERCICIOS NUMPY / ALGEBRA")
print("**************************")

# EJERCICIO 1. Importa numpy como np
import numpy as np

# EJERCICIO 2. Crea una lista de Python con los números del 1 a l0. Y después suma 5 a cada uno de sus elementos.
lista = list(range(1,11))
print([cada + 5 for cada in lista])

# EJERCICIO 3. Haz lo mismo pero ahora usando NumPy (primero crea un vector a partir de lista).
vector = np.array(lista)
print(vector + 5)

# EJERCICIO 4. Crea un array directamente desde un rango, desde el 100 al 200 de 10 en 10 (201 porque es abierto)
vector = np.arange(100,201,10)
print(vector)

# EJERCICIO 5. Crea una array de 20 elementos interpolados desde el 100 al 200 (aquí llega al 200, no es abierto)
vector = np.linspace(100,200,20)
print(vector)

# EJERCICIO 6. Crea una matriz de 4 filas y 6 columnas rellena con ceros.
matriz = np.zeros((4,6)) 
print(matriz)

# EJERCICIO 7. Crea una matriz identidad de 4 filas y 4 columnas.
matriz = np.eye(4) 
print(matriz)

# EJERCICIO 8. Crea una matriz de 6 filas y 3 columnas rellena con números aleatorios entre 0 y 1.
matriz = np.random.rand(6,3) # Con el método random de numpy
print(matriz)

# EJERCICIO 9. Crea un vector aleatorio de 100 elementos que tenga una distribución normal.

# Una distribución normal con media 0
vector = np.random.randn(100)
print(vector)

# EJERCICIO 10. Crea un vector aleatorio de 100 elementos que vayan entre el 1 y el 100.

# Una distribución uniforme (campana de Gauss) entre dos int
vector = np.random.randint(1,101,100) # 101 porque es un rango abierto
print(vector)

# EJERCICIO 11. Crea un vector llamado vector, como el anterior pero ordénalo.
vector.sort()
print(vector)

# EJERCICIO 12. Sobre vector cacula la media, mediana, desv típica, varianza, máximo y mínimo.

print(vector.mean()) # La media
print(np.median(vector)) # La mediana
print(np.std(vector)) # Desviación típica
print(vector.var()) # Varianza
print(vector.max()) # Máximo
print(vector.min()) # Mínimo

# EJERCICIO 13. Extrae de vector los elementos que están entre el sexto y el octavo ambos incluidos.

print(vector[5:8])

# EJERCICIO 14. Extrae de vector los elementos que son pares. Pista: puedes apoyarte en el operador módulo % (el resto que queda cuando haces una división)

pares = vector[vector % 2 == 0]
print(pares) # Me muestra un vector con los elementos pares

# EJERCICIO 15. Crea dos vectores de enteros llamados v1 y v2, cada uno de 5 elementos aleatorios entre 1 y 10 y súmalos.

v1 = np.array([1,2,3,4,5])
v2 = np.array([6,7,8,9,10])

# Para crearlos aleatorios np.random.randint(1,11,5)

print(v1 + v2)

# EJERCICIO 16. Multiplícalos usando el producto escalar.

v1 = np.array([1,2,3,4,5])
v2 = np.array([6,7,8,9,10])

print(np.dot(v1,v2))

# EJERCICIO 17. Transforma el vector que creaste antes llamado vector en una matriz de 10 x 10 y llámala matriz

print(vector.shape) # Comprobamos la dimensión del vector
matriz = vector.reshape(10,10)
print(matriz)

# EJERCICIO 18. Multiplica matriz por 8

print(matriz * 8)

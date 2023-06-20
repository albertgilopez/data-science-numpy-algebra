print("*******")
print("ALGEBRA")
print("*******")

print("*******************")
print("ESTRUCTURAS BÁSICAS")
print("*******************")
print("\n")

import numpy as np 

# Realmente no es necesario un conocimiento profundo de álgebra para hacer machine learning.
# Pero si es interesante por lo menos tener una nociones de lo que está pasando por debajo.
# Si te quieres dedicar a la investigación entonce si deberías aprender más en profundidad.

# En concreto, es importante conocer tres estructuras de datos:

# Escalares: en Python son los "datos individuales". Geométricamente tienen 1 punto y tienen dimensión 0
# Vectores: secuencias de varios escalares. Son una recta y tienen dimensión 1. Matrices de 1 dimensión, tal que, m x 1 (verticales) o 1 x m (horizontales)
# Matrices: secuencias de varios vectores organizados en dos dimensiones. Son un plano y tienen dimensión 2, que se denota por m x n

# Se puede seguir extendiendo a arrays multidimensionales, que son hiperplanos con dimension n.

# Ejemplo de escalar con NumPy:
e = np.array(1)
print(e)

# Ejemplo de vector en NumPy:
v = np.array([1,2,3])
print(v)

# Ejemplo de matriz en NumPy:
m = np.array([[1,2,3],[4,5,6]])
print(m)

# Ejemplo de matriz multidimensional en NumPy:
m = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
print(m)

print("\n")
print("************************")
print("OPERACIONES CON VECTORES")
print("************************")
print("\n")

# Algebra lo que significa es el conjunto de reglas para hacer las operaciones con ciertos tipos de elementos
# Como sumar o restar, tienen sus propias reglas aritméticas, p.e. sumar vectores, multiplicar vectores, matrices, etc.

# Las condiciones son que tienen que tener la misma longitud (se sum como en columnas 1 + 4, 2 + 5, 3 + 6)
v1 = np.array([1,2,3])
v2 = np.array([4,5,6])

print(v1 + v2) 

# Para restar vectores igual, las condiciones son que tienen que tener la misma longitud.
v1 = np.array([1,2,3])
v2 = np.array([4,5,6])

print(v1 - v2) 

# Para multiplicar un vector por un escalar, multiplicamos el escalar por cada elemento del vector
v1 = np.array([1,2,3])

print(v1 * 10)

# Para multiplicar dos vectores, tienen que tener la misma longitud.
# La multiplicación más común es el producto escalar o punto producto, esto es:

# Multiplicar uno a uno los correspondientes (como la suma y la resta) y luego sumar el resultado. 
v1 = np.array([1,2,3])
v2 = np.array([4,5,6])

print(np.dot(v1,v2)) 

print("\n")
print("************************")
print("OPERACIONES CON MATRICES")
print("************************")
print("\n")

# Las condiciones para la suma y la resta que tienen que tener la misma forma con su correspondiente

m1 = np.array([[3,-4,6],[0,-8,3]])
m2 = np.array([[-1,-9,7],[4,8,-7]])

print(m1 + m2)

m1 = np.array([[3,-4,6],[0,-8,3]])
m2 = np.array([[-1,-9,7],[4,8,-7]])

print(m1 - m2)

# La multiplicación por un escalar funciona igual que en los vectores, se multiplic por cada elemento

m1 = np.array([[3,-4,6],[0,-8,3]])

print(m1 * 10)

# Para multiplicar matrices usaremos el producto escalar
# La condición es que el número de columnas de la primera tiene que ser el mismo que el número de filas de la segunda, esto es multiplicar m x n por una n x k
# Es decir, multiplicar una 3 x 2 por una 2 x 4, dónde el resultado será una nueva matriz m x k, p.e. 3 x 4 (en el ejemplo anterior)
# Donde el producto escalar de fila 1 de la primera por columna 1 de la segunda será el resultado de la fila 1, columna 1 de la tercera

m1 = np.array([[3,-4,6],[0,-8,3]])
m2 = np.array([[-1,-9,7],[4,8,-7]])

print(m1.shape) # Comprobamos la dimensión de m1

# Y creamos otra matriz m3

m3 = np.array([
	[8,6,2],
	[6,3,4],
	[2,-4,6]
])

print(m3.shape) # Comprobamos la dimensión de m3

# Aplicamos la función dot para aplicar el producto escalar
print(np.dot(m1,m3))

# Multiplicamos los elementos de la fila 1 por los elementos de la columna 1
# Como hacíamos en la multiplicción de vectores, el resultado final de los elementos se suma

# En el ejemplo, al haber 3 columnas, faltaría multiplicar la fila 1 por la columna 2 y po la columna 3
# Luego, la fila 2, por la columna 1, columna 2 y column 3. El resultado, una matriz de 2 x 3


# Habrá ocasiones en machine learning que aunque tengamos que usar una matriz el formato en el que tenemos que pasarle esa matriz a la clase que la procesará es un array multidimensional
# Entonces, para aplanar una matriz (flatten en inglés) significa eso, pasarla de dimension 2 a dimension 1, básicamente concatenando sus filas en una sola, utilizamos este método en NumPy:

m3 = np.array([
	[8,6,2],
	[6,3,4],
	[2,-4,6]
])

print(m3.flatten())

# La mayoría de las operaciones en ML se pueden hacer mediante álgebra matricial. Es más eficiente computacionalmente
# En Deep Learning (subcampo de machine learning basado en la evolución de un algoritmo concreto llamado redes neuronales) o Computer Vision

# El algoritmo de redes neuronales ha evolucionado mucho, gracias a la facilidad de computación y a los volúmenes de los datos (sobre todo no estructurados: audio, imagen...)
# Para datos estructurados (los que tienen la mayoría de las empresas) deep learning no ha mostrado una mejora de rendimiento considerable

# Por ejemplo, TensorFlow es el framework de deep learning de Google
# Se llama así porque los tensores son la base del deep learning, una generalización de lo que ya conocemos:

# Un escalar es un tensor de rango 0
# Un vector es un tensor de rango 1
# Una matriz es un tensor de rango 2
# Un tensor de rango 3 sería una colección de matrices

print("\n")
print("*******************************")
print("DEEP LEARNING / COMPUTER VISION")
print("*******************************")
print("\n")

m1 = np.array([[3,-4,6],[0,-8,3]])
m2 = np.array([[-1,-9,7],[4,8,-7]])

tensor = np.array([m1,m2]) # Array de dimension 3

print(tensor)
print(tensor.shape) # 2 matrices de 2 filas x 3 columnas

# Pongamos el ejemplo de Computer Vision.

# En Deep Learning se trabaja mucho con imágenes. Pero un ordenador no ve las imágenes como nosotros.
# Cada imagen es en realidad un conjunto de pixeles, p.e. de 1000x1000, es decir una matriz, o un tensor de rango 2.
# En una imagen en blanco y negro cada celda sería un número entre 0 y 255 que denota la escala de gris. Siendo 0 blanco puro y 255 negro puro.
# Una forma de incluir color es usar las escalas RGB, donde cualquier color se puede representar como una combinación de esos 3 colores.
# Por tanto tendríamos una matriz de 1000x1000 por cada uno de esos 3 canales, es decir 1000x1000x3 o un tensor de rango 3.
# Y así es como podemos representar una imagen con álgebra matricial y aplicarle machine learning.

# Por ejemplo vamos a poner un caso simplificado donde la imagen fuera solo de 5x5 pixeles.
# Usando una función de NumPy para generar aleatorios entre 0 y 255 así es como el ordendador "vería" esa imagen.

R = np.random.randint(0,255,(5,5)) # Tonalidad de rojos
G = np.random.randint(0,255,(5,5)) # Tonalidad de verdes
B = np.random.randint(0,255,(5,5)) # Tonalidad de azules

imagen = np.array([R,G,B]) # Tensor de rango 3 (3,5,5) 3 matrices de 5 filas por 5 columnas

print(imagen)


# Para saber más: https://betterprogramming.pub/numpy-illustrated-the-visual-guide-to-numpy-3b1d4976de1d
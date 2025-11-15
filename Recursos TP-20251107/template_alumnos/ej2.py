import numpy
from dataset import cargarDataset
from pathlib import Path
from alc import *

data_path = Path("./dataset/cats_and_dogs")
X_train, Y_train, X_val, Y_val = cargarDataset(data_path)

print("Xt: ", X_train.shape)
print("Yt: ", Y_train.shape)
print("Xv: ", X_val.shape)
print("Yv: ", Y_val.shape)

X_train = X_train[:,:100]
Y_train = Y_train[:100,:]
print(X_train.shape)
print(Y_train.shape)
# Me queda con que L * L transpuesta (cholesky) * U = X transpuesta
# Lo copado de esto es que L es triangular inferior, y Lt es triangular superior
# Entonces resolver los sistemas es mas o menos sencillo por ej si tengo
# | a11 0    0  |   | u11 u12 u13 |   =   | x11 x12 x13 |
# | a21 a22 0   | * | u21 u22 u23 |   =   | x21 x22 x23 |
# | a31 a32 a33 |   | u31 u32 u33 |   =   | x31 x32 x33 |
# Puedo resolver primero para la primer fila de U,
# despues para la segunda fila de U usando la primer fila ya resuelta, y asi.
# Por ej yo ya se que:
# a22 * u21 = x21 - a21 * u11
# Entonces lo que hago es buscar solucion a L * B = X transpuesta para despues hacer
# U = solve(Lt, B)


# Hago esto para usar versiones mas rapidas que las que escribimos a manopla
transpose = numpy.transpose
solve = numpy.linalg.solve
cholesky = cholesky
#mul = lambda A, B: A @ B
mul = matMul

def caso1(X, Y, transpose, solve, cholesky, matmul):
    A = matmul(X, transpose(X))
    L, Lt = cholesky(A)
    B = solve(L, X)
    U = solve(Lt, B)
    W = matmul(U, Y)
    return W

def caso2(X, Y, transpose, solve, cholesky, matmul):
    A = matmul(X, transpose(X))
    L, Lt = cholesky(A)
    B = solve(L, X)
    U = solve(Lt, B)
    W = matmul(U, Y)
    return W

def connected_con_cholesky(X, Y):
    caso1(X, Y, transpose, solve, cholesky, matMulSlow)

#connected_con_cholesky(X_train, Y_train)

# Time matmulSlow vs matmul
import time
start = time.time()
W2 = matMulFast(X_train, transpose(X_train))
end = time.time()
print("matMulFast time: ", end - start)
start = time.time()
W = matMul(X_train, transpose(X_train))
end = time.time()
print("matMul time: ", end - start)
start = time.time()
W2 = matMulSlow(X_train, transpose(X_train))
end = time.time()
print("matMulSlow time: ", end - start)


#Compare results for correctnes
print(f"W: {W}, W2: {W2}")
W_diff = numpy.abs(W - W2)
print("Max difference between matMul and matMulSlow: ", numpy.max(W_diff))
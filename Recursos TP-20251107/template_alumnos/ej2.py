import numpy
from dataset import cargarDataset
from pathlib import Path
from alc import *

if '__name__' == '__main__':
    data_path = Path("./dataset/cats_and_dogs")
    X_train, Y_train, X_val, Y_val = cargarDataset(data_path)

    #X_train = numpy.concatenate((X_train[:,850:1150],X_train[:,850:1150]),axis=1)
    #Y_train = numpy.concatenate((Y_train[850:1150,:],Y_train[850:1150,:]),axis=0)

    # Transpose Y to match expected dimensions: (classes x samples) instead of (samples x classes)
    Y_train = transpuesta(Y_train)
    Y_val = transpuesta(Y_val)

    print("Xt: ", X_train.shape)
    print("Yt: ", Y_train.shape)
    print("Xv: ", X_val.shape)
    print("Yv: ", Y_val.shape)

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
transpose = transpuesta
solve = numpy.linalg.solve
cholesky = cholesky
#mul = lambda A, B: A @ B
mul = matMul

def caso1(X, Y, transpose, solve, cholesky, matmul):
    X_T = transpose(X)
    A = matmul(X_T, X)
    L, Lt = cholesky(A)
    # L * Lt * U = X_T
    # L * B = X_T
    B = solve(L, X_T)
    # Lt * U = B
    U = solve(Lt, B)
    # W = Y * U
    W = matmul(Y, U)
    return W


# Caso 2
# A = X * X transpuesta
# Quiero resolver V * A = X transpuesta
# Que es lo mismo que hacer A transpuesta * V transpuesta = X
# despejar W de W = Y * V
def caso2(X, Y, transpose, solve, cholesky, matmul):
    X_T = transpose(X)
    A = matmul(X, X_T)
    A_T = transpose(A)
    L, Lt = cholesky(A_T)
    B = solve(L, X)
    V_T = solve(Lt, B)
    V = transpose(V_T)
    W = matmul(Y, V)
    return W

# Despejar W de WX = Y
# W = Y * X+
def caso3(X, Y, transpose, solve, cholesky, matmul):
    W = matmul(Y, inversa(X))
    return W

"""
Dada X(n x p) ∈ R
     Y(m x p) ∈ R
1: a- Si rango(X) = p n > p
2: b- Si rango(X) = n n < p
3: c- Si rango(X) = n y n = p
4: Calcular W = Y X+
"""

def connected_con_cholesky(X, Y):
    Q,R = QR_con_GS(X)
    rank = rango_R(R)
    n = X.shape[0]
    p = X.shape[1]

    print("Rango de X:", rank)

    if rank == p and n > p:
        W = caso1(X, Y, transpose, solve, cholesky, matMul)
    elif rank == n and n < p:
        W = caso2(X, Y, transpose, solve, cholesky, matMul)
    elif rank == n and n == p:
        W = caso3(X, Y, transpose, solve, cholesky, matMul)
    else:
        raise ValueError("Rango de X no compatible con Y")
    return W

pinvEcuacionesNormales = connected_con_cholesky
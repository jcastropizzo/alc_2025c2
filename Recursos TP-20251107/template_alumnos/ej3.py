from dataset import cargarDataset
from pathlib import Path
from alc import *
import numpy as np
import time



def inversa_diagonal(A):
    # 1. Create a writeable copy of the array A
    B = A.copy()
    for i in range(B.shape[0]):
        if B[i,i] != 0:
            B[i,i] = 1/B[i,i]
    return B

def pinvSVD(U, S, V, Y):
    n = U.shape[0]

    Ut = transpuesta(U)
    V1 = V[:,0:n]
    St = inversa_diagonal(S)
    W = matMul(matMul(matMul(transpuesta(Y),V1),St),Ut)
    return W
from dataset import cargarDataset
from pathlib import Path
from alc import *
import numpy as np
import time

main_time_start = time.perf_counter()

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

    print("calculando W...")
    W_start_time = time.perf_counter()
    W = matMul(matMul(matMul(transpuesta(Y),V1),St),Ut)
    print(f"W calculado en : {time.perf_counter() - W_start_time:.4f} sec")
    main_time_end = time.perf_counter()
    elapsed = main_time_end - main_time_start
    print(f"Elapsed time: {elapsed:.4f} sec")
    return W
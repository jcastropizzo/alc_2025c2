from dataset import cargarDataset
from pathlib import Path
import numpy as np


# Funcion que calcula la pseudoinversa de una matriz usando el metodo de Householder
# Q: matriz de Householder (matriz ortogonal)
# R: matriz triangular superior (matriz triangular superior)
# Y: matriz de datos (matriz de datos)
# Devuelve la pseudoinversa de la matriz (matriz de datos)
def pinvHouseHolder(Q: np.ndarray, R: np.ndarray, Y: np.ndarray) -> np.ndarray:
    # Algorithm 3, paso 3
    X_plus = Q @ R.T.inv()
    # Algorithm 3, paso 4-5
    # Como V = X_plus y V @ R.T = Q, entonces V = X_plus = Q @ R.T.inv()
    # W = Y @ V = Y @ X_plus
    W = Y @ X_plus
    return W

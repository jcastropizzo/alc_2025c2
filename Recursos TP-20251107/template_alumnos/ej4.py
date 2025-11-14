from dataset import cargarDataset
from pathlib import Path
import numpy as np
from alc import inversa, matMul, QR_con_GS, transpuesta


def _calcular_pesos_con_qr(Q: np.ndarray, R: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Implementación del Algoritmo 3 para calcular la matriz de pesos W.

    Este algoritmo resuelve el problema de mínimos cuadrados usando la pseudoinversa
    calculada a partir de una descomposición QR (independientemente del método usado).

    Args:
        Q: Matriz ortogonal de la descomposición QR
        R: Matriz triangular superior de la descomposición QR
        Y: Matriz de targets de entrenamiento (etiquetas/valores objetivo)

    Returns:
        W: Matriz de pesos que minimiza el error cuadrático medio

    Algoritmo 3:
        - Paso 3: Calcular X_plus = Q @ inv(R^T)
        - Pasos 4-5: Calcular W = Y @ X_plus
    """
    # Calcular la pseudoinversa: X_plus = Q @ inv(R^T)
    X_plus = matMul(Q, inversa(transpuesta(R)))

    # Calcular los pesos: W = Y @ X_plus
    W = matMul(Y, X_plus)

    return W


# 4.1
def pinvHouseHolder(Q: np.ndarray, R: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return _calcular_pesos_con_qr(Q, R, Y)


# 4.2
def pinvGramSchmidt(Q: np.ndarray, R: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return _calcular_pesos_con_qr(Q, R, Y)


def alc_imp_gs():
    X_train, Y_train, X_val, Y_val = cargarDataset(Path("./dataset/cats_and_dogs"))

    Q, R = QR_con_GS(X_train)
    return pinvGramSchmidt(Q, R, Y_train)

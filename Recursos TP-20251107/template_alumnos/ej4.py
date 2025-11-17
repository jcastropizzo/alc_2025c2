from dataset import cargarDataset
from pathlib import Path
import numpy as np
import os
from alc import QR_con_HH, inversa, matMul, QR_con_GS, transpuesta, debugPrint


def _calcular_pesos_con_qr(Q: np.ndarray, R: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Implementación del Algoritmo 3 para calcular la matriz de pesos W.

    Args:
        Q: Matriz con columnas ortonormales de la descomposición QR (n x n)
        R: Matriz triangular superior de la descomposición QR (n x p)
        Y: Matriz de targets de entrenamiento (n x k) donde n = n_classes y k = n_samples
 
    Returns:
        W: Matriz de pesos (n x k)

    Algoritmo 3:
        Para matriz X (n x p) con n < p:

        Derivación de la fórmula simplificada:
        1. Pseudoinversa para matriz X (n x p) con n < p:
           X⁺ = X^T @ (X @ X^T)^{-1}

        2. Usar la descomposición QR de X^T = Q @ R.
           Entonces X^T = Q @ R
           Por lo tanto, X = (Q @ R)^T = R^T @ Q^T

        3. Calcular X @ X^T:
           X @ X^T = (R^T @ Q^T) @ (Q @ R)
                   = R^T @ (Q^T @ Q) @ R
           Como Q tiene columnas ortonormales, Q^T @ Q = I,
           así que X @ X^T = R^T @ R

        4. Calcular la inversa:
           (X @ X^T)^{-1} = (R^T @ R)^{-1}

        5. Sustituir en la expresión de la pseudoinversa:
           X⁺ = X^T @ (X @ X^T)^{-1}
              = Q @ R @ (R^T @ R)^{-1}

        6. Utilizar la propiedad de la inversa de un producto:
           (R^T @ R)^{-1} = R^{-1} @ (R^T)^{-1}
           Entonces:
           X⁺ = Q @ R @ R^{-1} @ (R^T)^{-1}
              = Q @ I @ (R^T)^{-1}
              = Q @ (R^T)^{-1}

        7. Por lo tanto, X⁺ = Q @ (R^T)^{-1}
                   
        8. Calcular pesos:
           W = Y @ X⁺ = Y @ Q @ (R^T)^{-1}
    """
    debugPrint("[DEBUG] _calcular_pesos_con_qr: Iniciando cálculo")
    debugPrint(f"[DEBUG] _calcular_pesos_con_qr: Forma Q: {Q.shape}, Forma R: {R.shape}, Forma Y: {Y.shape}")
    
    # El desarrollo muestra que la pseudoinversa para X (n x p) con n < p, usando QR de X^T = Q @ R, da:
    # X⁺ = Q @ (R^T)^{-1}
    # Donde Q: (p x n), R: (n x n)
    # Por tanto, W = X⁺ @ Y = Q @ (R^T)^{-1} @ Y

    # (Q es (p x n), Y es (n x k), W será (p x k))

    # (R^T)^{-1} = inversa(transpuesta(R))
    debugPrint("[DEBUG] _calcular_pesos_con_qr: Calculando transpuesta de R")
    R_T = transpuesta(R)
    
    debugPrint("[DEBUG] _calcular_pesos_con_qr: Calculando inversa de R_T")
    R_T_inv = inversa(R_T)

    # X⁺ = Q @ (R^T)^{-1}
    debugPrint("[DEBUG] _calcular_pesos_con_qr: Calculando X_plus = Q @ R_T_inv")
    X_plus = matMul(Q, R_T_inv)

    # W = Y @ X⁺ (según línea 54 del desarrollo matemático)
    debugPrint("[DEBUG] _calcular_pesos_con_qr: Calculando W = Y @ X_plus")
    W = matMul(Y, X_plus)
    
    debugPrint(f"[DEBUG] _calcular_pesos_con_qr: Completado. Forma W: {W.shape}")
    return W


# 4.1
def pinvHouseHolder(Q: np.ndarray, R: np.ndarray, Y: np.ndarray) -> np.ndarray:
    debugPrint(f"[DEBUG] pinvHouseHolder: Llamada con Q forma: {Q.shape}, R forma: {R.shape}, Y forma: {Y.shape}")
    return _calcular_pesos_con_qr(Q, R, Y)


# 4.2
def pinvGramSchmidt(Q: np.ndarray, R: np.ndarray, Y: np.ndarray) -> np.ndarray:
    debugPrint(f"[DEBUG] pinvGramSchmidt: Llamada con Q forma: {Q.shape}, R forma: {R.shape}, Y forma: {Y.shape}")
    return _calcular_pesos_con_qr(Q, R, Y)


def alc_imp_gs():
    debugPrint("[DEBUG] alc_imp_gs: Iniciando implementación con Gram-Schmidt")
    
    debugPrint("[DEBUG] alc_imp_gs: Cargando dataset")
    X_train, Y_train, X_val, Y_val = cargarDataset(Path("./dataset/cats_and_dogs"))
    
    debugPrint("="*50)
    debugPrint("GRAM-SCHMIDT")
    debugPrint("="*50)
    debugPrint(f"Original X_train shape: {X_train.shape}")
    debugPrint(f"Original Y_train shape: {Y_train.shape}")

    debugPrint(f"[DEBUG] alc_imp_gs: Dataset cargado - Forma X_train: {X_train.shape}, Forma Y_train: {Y_train.shape}")
    
    # X_train ya es (n_features, n_samples) = (1536, 2000) = (n, p) con n < p (wide) ✓
    # Y_train es (n_samples, n_classes) = (2000, 2), pero el algoritmo necesita (k, p) = (2, 2000)
    # Transponer Y para obtener (n_classes, n_samples) = (k, p)
    debugPrint("[DEBUG] alc_imp_gs: Transponiendo Y_train para obtener (k, p)")
    Y_train = transpuesta(Y_train)
    
    # Algoritmo requiere QR de X^T donde X es (n, p) con n < p
    # X es (1536, 2000), así que X^T es (2000, 1536) tall
    debugPrint("[DEBUG] alc_imp_gs: Calculando descomposición QR con Gram-Schmidt de X^T")
    X_train_T = transpuesta(X_train)
    Q, R = QR_con_GS(X_train_T)
    
    debugPrint(f"X_train shape: {X_train.shape}")
    debugPrint(f"X_train_T shape: {X_train_T.shape}")
    debugPrint(f"Y_train shape: {Y_train.shape}")
    debugPrint(f"Q shape: {Q.shape}")
    debugPrint(f"R shape: {R.shape}")
    
    debugPrint("[DEBUG] alc_imp_gs: Calculando pesos usando pinvGramSchmidt")
    W = pinvGramSchmidt(Q, R, Y_train)
    debugPrint(f"W shape: {W.shape}")
    
    debugPrint("[DEBUG] alc_imp_gs: Completado exitosamente")
    # validate_transferlearning(W, X_val, Y_val)
    # matriz_confusion(W, X_val, Y_val)
    return 1


def alc_imp_hh():
    debugPrint("[DEBUG] alc_imp_hh: Iniciando implementación con Householder")
    
    debugPrint("[DEBUG] alc_imp_hh: Cargando dataset")
    X_train, Y_train, X_val, Y_val = cargarDataset(Path("./dataset/cats_and_dogs"))
    
    debugPrint("="*50)
    debugPrint("HOUSEHOLDER")
    debugPrint("="*50)
    debugPrint(f"Original X_train shape: {X_train.shape}")
    debugPrint(f"Original Y_train shape: {Y_train.shape}")

    debugPrint(f"[DEBUG] alc_imp_hh: Dataset cargado - Forma X_train: {X_train.shape}, Forma Y_train: {Y_train.shape}")
    
    # X_train ya es (n_features, n_samples) = (1536, 2000) = (n, p) con n < p (wide) ✓
    # Y_train es (n_samples, n_classes) = (2000, 2), pero el algoritmo necesita (k, p) = (2, 2000)
    # Transponer Y para obtener (n_classes, n_samples) = (k, p)
    debugPrint("[DEBUG] alc_imp_hh: Transponiendo Y_train para obtener (k, p)")
    Y_train = transpuesta(Y_train)
    
    # Algoritmo requiere QR de X^T donde X es (n, p) con n < p
    # X es (1536, 2000), así que X^T es (2000, 1536) tall
    debugPrint("[DEBUG] alc_imp_hh: Calculando descomposición QR con Householder de X^T")
    X_train_T = transpuesta(X_train)
    Q, R = QR_con_HH(X_train_T)
    
    debugPrint(f"X_train shape: {X_train.shape}")
    debugPrint(f"X_train_T shape: {X_train_T.shape}")
    debugPrint(f"Y_train shape: {Y_train.shape}")
    debugPrint(f"Q shape: {Q.shape}")
    debugPrint(f"R shape: {R.shape}")
    
    debugPrint("[DEBUG] alc_imp_hh: Calculando pesos usando pinvHouseHolder")
    W = pinvHouseHolder(Q, R, Y_train)
    debugPrint(f"W shape: {W.shape}")
    
    debugPrint("[DEBUG] alc_imp_hh: Completado exitosamente")
    # validate_transferlearning(W, X_val, Y_val)
    # matriz_confusion(W, X_val, Y_val)
    return 1


alc_imp_gs()

alc_imp_hh()
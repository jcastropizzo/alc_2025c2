from alc import *
import numpy as np
from dataset import cargarDataset
from pathlib import Path
from ej2 import pinvEcuacionesNormales
from ej3 import pinvSVD
from ej4 import pinvHouseHolder, pinvGramSchmidt
from ej5 import esPseudoInversa
import time

def matriz_confusion(W, X_val, Y_val):
    conf_mat = np.array([[0,0],[0,0]])

    samples = X_val.shape[1]
    for i in range(samples):
        y_true_vector = Y_val[i, :]
        ei = W @ X_val[:, i]
        predicted_class = np.argmax(ei)
        true_class = np.argmax(y_true_vector)
        conf_mat[true_class, predicted_class] += 1

    verdaderos_gatos = conf_mat[0, 0]
    falsos_perros = conf_mat[0, 1]  # Gatos Reales, predichos como Perros
    falsos_gatos = conf_mat[1, 0]  # Perros Reales, predichos como Gatos
    verdaderos_perros = conf_mat[1, 1]

    print("\n" + "--- Matriz de Confusión (Validación) ---".center(45))
    print(" " * 15 + "Predicción: GATO | Predicción: PERRO |")
    print(" " * 17 + "-" * 37)

    # F-string con formato:
    # :<14 alinea el texto a la izquierda en un espacio de 14 chars
    # :^15 centra el número en un espacio de 15 chars

    print(f"Realidad: GATO  | {verdaderos_gatos:^15} | {falsos_perros:^17} |")
    print(" " * 17 + "-" * 37)
    print(f"Realidad: PERRO | {falsos_gatos:^15} | {verdaderos_perros:^17} |")
    print(" " * 17 + "-" * 37)
    print("\n")

def validate_transferlearning(W, X_val, Y_val):
    predicciones_correctas = 0
    samples = X_val.shape[1]
    for i in range(samples):
        y_true_vector = Y_val[i, :]
        ei = W @ X_val[:, i]
        predicted_class = np.argmax(ei)
        true_class = np.argmax(y_true_vector)
        if predicted_class == true_class:
            predicciones_correctas += 1
        # print("iteration", i, ":",ei)
        # print("iteration", i, " Y",Y_val[i,:])

    accuracy = (predicciones_correctas / samples) * 100
    print(f"\n--- Resultados de Validación ---")
    print(f"Precisión (Accuracy): {accuracy:.2f}%")
    print(f"Clasificó correctamente {predicciones_correctas} de {samples} muestras.")
    return accuracy


def benchmark():
    X_train, Y_train, X_val, Y_val = cargarDataset(Path("./dataset/cats_and_dogs"))

    #Cholesky
    start_time = time.perf_counter()
    W = pinvEcuacionesNormales(X_train, Y_train)
    end_time = time.perf_counter()
    Cholesky_time = end_time - start_time
    print(f"Cholesky exercise executed in: {Cholesky_time:.4f} seconds")
    Cholesky_accuracy = validate_transferlearning(W,X_val,Y_val)
    matriz_confusion(W, X_val, Y_val)

    #SVD
    start_time = time.perf_counter()
    U, srow, V = np.linalg.svd(X_train)
    S = np.diag(srow)
    W_SVD = pinvSVD(U, S, V, Y_train,'np')
    end_time = time.perf_counter()
    SVD_time = end_time - start_time
    print(f"SVD exercise executed in: {SVD_time:.4f} seconds")
    SVD_accuracy = validate_transferlearning(W_SVD,X_val,Y_val)
    matriz_confusion(W_SVD, X_val, Y_val)

    #QR
    start_time = time.perf_counter()
    Q, R = QR_con_GS(X_train)
    W_GS =  pinvGramSchmidt(Q, R, Y_train)
    end_time = time.perf_counter()
    GS_time = end_time - start_time
    print(f"GS exercise executed in: {GS_time:.4f} seconds")
    GS_accuracy = validate_transferlearning(W_GS,X_val,Y_val)
    matriz_confusion(W_GS, X_val, Y_val)

    print("\n" + "--- Tablas de resultados ---".center(45))
    print(" " * 15 + " Tiempo | Accuracy |")
    print(" " * 17 + "-" * 37)
    print(f"SVD  | {SVD_time:^15} | {SVD_accuracy:^17} |")
    print(" " * 17 + "-" * 37)
    print(f"GS | {GS_time:^15} | {GS_accuracy:^17} |")
    print(" " * 17 + "-" * 37)
    print(f"CHL | {Cholesky_time:^15} | {Cholesky_accuracy:^17} |")
    print(" " * 17 + "-" * 37)
    print("\n")

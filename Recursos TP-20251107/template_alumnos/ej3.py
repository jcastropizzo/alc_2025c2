from dataset import cargarDataset
from pathlib import Path
from alc import *
import numpy as np
import sys
import argparse
from ej6 import *

def pinvSVD(U, S, V, Y,imp='alc'):
    n = U.shape[0]

    if imp == 'alc':
        Ut = transpuesta(U)
        Vt = transpuesta(V) # En realidad SVD devuelve V transpuesta, por lo cual usamos V en la inversa de X. Mantengo naming conventions del enunciado.
        V1 = Vt[:,0:n]
        S = inversa(S)
        return matMul(matMul(matMul(transpuesta(Y),V1),S),Ut)

    elif imp == 'np':
        Ut = U.T
        Vt = V.T # En realidad SVD devuelve V transpuesta, por lo cual usamos V en la inversa de X. Mantengo naming conventions del enunciado.
        V1 = Vt[:,0:n]
        S = np.linalg.pinv(S)
        return Y.T@V1@S@Ut
    return None

def alc_imp():
    X_train, Y_train, X_val, Y_val = cargarDataset(Path("./dataset/cats_and_dogs"))
    U, S, V = svd_reducida(X_train)
    W = pinvSVD(U, S, V, Y_train,'alc')
    validate_transferlearning(W,X_val,Y_val)
    matriz_confusion(W, X_val, Y_val)

def np_imp():
    X_train, Y_train, X_val, Y_val = cargarDataset(Path("./dataset/cats_and_dogs"))

    U, srow, V = np.linalg.svd(X_train)
    S = np.diag(srow)
    W = pinvSVD(U, S, V, Y_train,'np')

    validate_transferlearning(W,X_val,Y_val)
    matriz_confusion(W, X_val, Y_val)
    return 1

# # --- Setup Argument Parser ---
# parser = argparse.ArgumentParser(
#     description="Process data based on the mode: 'np' (Non-Photometric) or 'alc' (Alkali Content)."
# )
#
# # Define the positional argument named 'mode'
# parser.add_argument(
#     "mode",
#     choices=["np", "alc"],  # This restricts the input to only these two strings
#     help="The desired execution mode: 'np' or 'alc'."
# )
#
# # --- Parse and Handle Input ---
# try:
#     args = parser.parse_args()
#
#     # Use an if/elif/else structure to branch based on the parameter's value
#     if args.mode == "np":
#         print("Corriendo implementación de NumPy...")
#         np_imp()
#     elif args.mode == "alc":
#         print("Corriendo implementación de ALC...")
#         alc_imp()
#
# except SystemExit:
#     # argparse raises SystemExit if the input is invalid (e.g., neither 'np' nor 'alc')
#     # and prints the automatic help message.
#     sys.exit(1)


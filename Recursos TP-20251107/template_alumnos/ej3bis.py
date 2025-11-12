from dataset import cargarDataset
from pathlib import Path
from alc import *
import numpy as np

data_path = Path("./dataset/cats_and_dogs")
X_train, Y_train, X_val, Y_val = cargarDataset(data_path)

print("Xt: ", X_train.shape)
print("Yt: ", Y_train.shape)
print("Xv: ", X_val.shape)
print("Yv: ", Y_val.shape)

U, S, V = np.linalg.svd(X_train)

print("U: ",U.shape)
print("S: ",S.shape)
print("V: ",V.shape)

def pinvSVD(U, S, V, Y):

    n = U.shape[0]
    p = V.shape[0]
    s1_inv = np.zeros_like(s, dtype=float)
    non_zero_indices = s > 1e-15
    s_inv[non_zero_indices] = 1.0 / s[non_zero_indices]
    splus = np.pad(s_inv, (0, p), 'constant')
    sigmaplus = np.diag(splus)   
    Ut = transpuesta(U)
    Vt = transpuesta(Vt) # En realidad SVD devuelve V transpuesta, por lo cual usamos V en la inversa de X. Mantengo naming conventions del enunciado.
    V1 = Vt[:,0:n]
    V2 = Vt[:n:p]
    
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

U, srow, V = np.linalg.svd(X_train)
S = np.diag(srow)
print("U: ",U.shape)
print("S: ",S.shape)
print("V: ",V.shape)

def pinvSVD(U, S, V, Y):

    n = U.shape[0]
    p = V.shape[0]
    print("n: ", n)
    print("p :", p)
    print("p - n: ", p-n)
    
    
    Ut = transpuesta(U)
    Vt = transpuesta(V) # En realidad SVD devuelve V transpuesta, por lo cual usamos V en la inversa de X. Mantengo naming conventions del enunciado.
    V1 = Vt[:,0:n]
    V2 = Vt[:,n:p]
    
    print("V1 shape", V1.shape)
    print("V2 shape", V2.shape)
    VSigmamas = V1@S
    print("Vsigmamas shape", VSigmamas.shape)
    # luego Vsigmamas@Ut
    VSigmamasUt = VSigmamas@Ut
    print("VSmasUt shape", VSigmamasUt.shape)
    W = Y_train.T@VSigmamas@Ut
    return W

pinvSVD(U,S,V,Y_train)
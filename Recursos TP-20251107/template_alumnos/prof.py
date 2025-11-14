from alc import *
import numpy as np

X = np.random.rand(470, 490)

U, S, V = svd_reducida(X)
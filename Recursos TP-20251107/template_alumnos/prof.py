from alc import *
import numpy as np

X = np.random.rand(100, 150)

U, S, V = svd_reducida(X)
from dataset import cargarDataset
from pathlib import Path

data_path = Path("./dataset/cats_and_dogs")
X_train, Y_train, X_val, Y_val = cargarDataset(data_path)

print(X_train.shape)
print(Y_train.shape)
print(X_val.shape)
print(Y_val.shape)

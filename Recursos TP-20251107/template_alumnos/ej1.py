from dataset import cargarDataset
from pathlib import Path

data_path = Path('./dataset/cats_and_dogs')
X, Y = cargarDataset(data_path)
from pathlib import Path
from numpy import (
    load, ndarray, concatenate, zeros, ones, column_stack
)

CLASSES = ['cats', 'dogs']
EMBEDDING_NAME = 'efficientnet_b3_embeddings.npy'

Datos = tuple[ndarray, ndarray]

def cargarDataset(file: Path) -> Datos:
    train_path = file / 'train'
    paths = [train_path / cls / EMBEDDING_NAME for cls in CLASSES]
    training_data = concatenate([load(path) for path in paths], axis=1)
    label_data = column_stack([ones(training_data.shape[1] // 2), zeros(training_data.shape[1] // 2)])
    label_data = concatenate([label_data, column_stack([zeros(training_data.shape[1] // 2), ones(training_data.shape[1] // 2)])], axis=0)

    assert label_data.shape[0] == training_data.shape[1]
    assert training_data.shape[0] == 1536
    return training_data, label_data
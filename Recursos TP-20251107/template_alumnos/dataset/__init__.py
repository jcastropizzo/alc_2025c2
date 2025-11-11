from pathlib import Path
from numpy import load, ndarray, concatenate, zeros, ones, column_stack, array

CLASSES = ["cats", "dogs"]
EMBEDDING_NAME = "efficientnet_b3_embeddings.npy"

Datos = tuple[ndarray, ndarray, ndarray, ndarray]


def cargarDataset(file: Path) -> Datos:
    val_path = file / "val"
    train_path = file / "train"
    train_paths = [train_path / cls / EMBEDDING_NAME for cls in CLASSES]
    val_paths = [val_path / cls / EMBEDDING_NAME for cls in CLASSES]

    training_data = concatenate([load(path) for path in train_paths], axis=1)
    training_labels = array([[1, 0]] * (training_data.shape[1] // 2))
    training_labels = concatenate(
        [training_labels, array([[0, 1]] * (training_data.shape[1] // 2))], axis=0
    )

    validation_data = concatenate([load(path) for path in val_paths], axis=1)
    validation_labels = array([[1, 0]] * (validation_data.shape[1] // 2))
    validation_labels = concatenate(
        [validation_labels, array([[0, 1]] * (validation_data.shape[1] // 2))],
        axis=0,
    )

    return training_data, training_labels, validation_data, validation_labels

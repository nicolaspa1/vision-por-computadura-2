#!/usr/bin/env python3
"""
Utilidades para el proyecto de detección de grietas
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import numpy as np
import random
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import time


def set_seed(seed=42):
    """Fijar semilla para reproducibilidad"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Para mayor determinismo (puede ser más lento)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CrackDataset(Dataset):
    """Dataset personalizado para grietas en concreto"""

    def __init__(self, data_dir, transform=None, split='train'):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split

        # Cargar rutas de imágenes
        self.image_paths = []
        self.labels = []

        # Clase 0: Sin grieta (Negative)
        negative_dir = self.data_dir / "Negative"
        if negative_dir.exists():
            for img_path in negative_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.image_paths.append(img_path)
                    self.labels.append(0)

        # Clase 1: Con grieta (Positive)
        positive_dir = self.data_dir / "Positive"
        if positive_dir.exists():
            for img_path in positive_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.image_paths.append(img_path)
                    self.labels.append(1)

        print(f"Dataset {split}: {len(self.image_paths)} imágenes")
        print(f"  - Sin grieta: {self.labels.count(0)}")
        print(f"  - Con grieta: {self.labels.count(1)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from PIL import Image

        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Cargar imagen
        image = Image.open(img_path).convert('RGB')

        # Aplicar transformaciones
        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(input_size=224, augment=True):
    """Obtener transformaciones para train y validación"""

    # Transformaciones básicas
    base_transforms = [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]

    # Transformaciones con augmentation para entrenamiento
    if augment:
        train_transforms = transforms.Compose([
            transforms.Resize((input_size, input_size)),

            # Augmentaciones geométricas
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),

            # Augmentaciones de color
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomGrayscale(p=0.1),

            # Transformaciones adicionales
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),

            # Noise y dropout espacial
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
        ])
    else:
        train_transforms = transforms.Compose(base_transforms)

    # Transformaciones para validación (sin augmentation)
    val_transforms = transforms.Compose(base_transforms)

    return train_transforms, val_transforms


def get_data_loaders(data_dir, batch_size=32, val_split=0.2, test_split=0.1,
                     input_size=224, num_workers=0, augment=True):
    """Crear data loaders con splits estratificados"""

    # Obtener transformaciones
    train_transforms, val_transforms = get_transforms(input_size, augment)

    # Crear dataset completo con transformaciones de validación
    full_dataset = datasets.ImageFolder(data_dir, transform=val_transforms)

    # Calcular tamaños de splits
    total_size = len(full_dataset)
    test_size = int(test_split * total_size)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size - test_size

    print(f"📊 División del dataset:")
    print(f"  - Entrenamiento: {train_size} ({train_size / total_size:.1%})")
    print(f"  - Validación: {val_size} ({val_size / total_size:.1%})")
    print(f"  - Test: {test_size} ({test_size / total_size:.1%})")

    # Split estratificado manual
    set_seed(42)  # Para reproducibilidad

    # Obtener índices por clase
    class_indices = {0: [], 1: []}
    for idx, (_, label) in enumerate(full_dataset.samples):
        class_indices[label].append(idx)

    # Mezclar índices
    for class_idx in class_indices.values():
        random.shuffle(class_idx)

    # Dividir cada clase proporcionalmente
    train_indices, val_indices, test_indices = [], [], []

    for label, indices in class_indices.items():
        n_total = len(indices)
        n_test = int(test_split * n_total)
        n_val = int(val_split * n_total)
        n_train = n_total - n_test - n_val

        test_indices.extend(indices[:n_test])
        val_indices.extend(indices[n_test:n_test + n_val])
        train_indices.extend(indices[n_test + n_val:])

    # Crear subsets
    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # Para Windows, simplificamos las transformaciones
    # El dataset ya tiene las transformaciones básicas aplicadas

    # Crear data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False  # Cambiar a False para CPU
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, val_loader, test_loader


def calculate_metrics(y_true, y_pred, y_probs=None):
    """Calcular métricas de evaluación comprehensivas"""

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted'),
    }

    # AUC solo si tenemos probabilidades
    if y_probs is not None:
        try:
            if len(np.unique(y_true)) == 2:  # Clasificación binaria
                metrics['auc'] = roc_auc_score(y_true, y_probs[:, 1])
            else:
                metrics['auc'] = roc_auc_score(y_true, y_probs, multi_class='ovr')
        except:
            metrics['auc'] = 0.0

    return metrics


def save_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Guardar checkpoint del modelo"""

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'model_name': model.__class__.__name__
    }

    torch.save(checkpoint, filepath)
    print(f"✅ Checkpoint guardado: {filepath}")


def load_checkpoint(model, optimizer, filepath, device='cpu'):
    """Cargar checkpoint del modelo"""

    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', 0.0)
    accuracy = checkpoint.get('accuracy', 0.0)

    print(f"✅ Checkpoint cargado: {filepath}")
    print(f"   Época: {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    return epoch, loss, accuracy


class EarlyStopping:
    """Early stopping para evitar overfitting"""

    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.best_score = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()

        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()

        return False


class Timer:
    """Timer para medir tiempos de ejecución"""

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()
        return self.elapsed()

    def elapsed(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0

    def format_time(self, seconds):
        """Formatear tiempo en horas:minutos:segundos"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"


def get_device():
    """Obtener el mejor device disponible"""

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Usando GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memoria disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("💻 Usando CPU")

    return device


def create_results_dirs():
    """Crear directorios para resultados"""

    dirs = [
        'results/models',
        'results/plots',
        'results/logs',
        'results/checkpoints'
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    """Test de utilidades"""

    print("🔧 Probando utilidades...")
    print("=" * 50)

    # Test device
    device = get_device()

    # Test seed
    set_seed(42)
    print("✅ Semilla fijada")

    # Test timer
    timer = Timer()
    timer.start()
    time.sleep(0.1)
    elapsed = timer.stop()
    print(f"✅ Timer: {timer.format_time(elapsed)}")

    # Test directorios
    create_results_dirs()
    print("✅ Directorios creados")

    print("\n🎉 Test de utilidades completado!")
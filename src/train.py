#!/usr/bin/env python3
"""
Pipeline de entrenamiento para detección de grietas en concreto
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
from pathlib import Path
import time
import json

from utils import (
    set_seed, get_device, get_data_loaders, calculate_metrics,
    save_checkpoint, EarlyStopping, Timer, create_results_dirs
)
from models import get_model


class CrackTrainer:
    """Entrenador para modelos de detección de grietas"""

    def __init__(self, model_name, data_dir='data/raw', config=None):

        # Configuración por defecto
        self.default_config = {
            'batch_size': 32,
            'learning_rate': 1e-3,
            'epochs': 20,
            'val_split': 0.2,
            'test_split': 0.1,
            'input_size': 224,
            'num_workers': 0,  # Cambiado para Windows
            'patience': 7,
            'min_lr': 1e-6,
            'weight_decay': 1e-4,
            'augment': True,
            'scheduler': 'cosine',
            'label_smoothing': 0.1
        }

        # Actualizar configuración
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)

        # Setup básico
        self.model_name = model_name
        self.data_dir = data_dir
        self.device = get_device()

        # Crear directorios
        create_results_dirs()

        # Inicializar componentes
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        # Historia de entrenamiento
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'learning_rates': []
        }

        # Timer
        self.timer = Timer()

        print(f"🎯 Configurando entrenador para {model_name}")
        print(f"📊 Configuración: {self.config}")

    def setup_model(self):
        """Configurar modelo, optimizador y scheduler"""

        # Crear modelo
        self.model = get_model(self.model_name, num_classes=2, pretrained=True)
        self.model = self.model.to(self.device)

        # Criterio de pérdida con label smoothing
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.config['label_smoothing']
        )

        # Optimizador
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999)
        )

        # Scheduler
        if self.config['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs'],
                eta_min=self.config['min_lr']
            )
        elif self.config['scheduler'] == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                min_lr=self.config['min_lr']
            )

        print(f"✅ Modelo configurado: {self.model.__class__.__name__}")

        # Resumen del modelo
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"📊 Parámetros: {total_params:,} total, {trainable_params:,} entrenables")

    def load_data(self):
        """Cargar datos"""

        print(f"📁 Cargando datos desde {self.data_dir}")

        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(
            data_dir=self.data_dir,
            batch_size=self.config['batch_size'],
            val_split=self.config['val_split'],
            test_split=self.config['test_split'],
            input_size=self.config['input_size'],
            num_workers=self.config['num_workers'],
            augment=self.config['augment']
        )

        print(f"✅ Datos cargados:")
        print(f"   - Train: {len(self.train_loader.dataset)} muestras")
        print(f"   - Val: {len(self.val_loader.dataset)} muestras")
        print(f"   - Test: {len(self.test_loader.dataset)} muestras")

    def train_epoch(self):
        """Entrenar una época"""

        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training', leave=False)

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Estadísticas
            running_loss += loss.item()
            predicted = output.argmax(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Actualizar progress bar
            pbar.set_postfix({
                'Loss': f'{running_loss / (batch_idx + 1):.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def validate_epoch(self):
        """Validar una época"""

        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation', leave=False)

            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                running_loss += loss.item()
                predicted = output.argmax(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                # Guardar para métricas
                probs = F.softmax(output, dim=1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                pbar.set_postfix({
                    'Loss': f'{running_loss / (len(all_preds) // self.config["batch_size"] + 1):.4f}',
                    'Acc': f'{100. * correct / total:.2f}%'
                })

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = correct / total

        # Calcular métricas adicionales
        metrics = calculate_metrics(
            all_targets, all_preds, np.array(all_probs)
        )

        return epoch_loss, epoch_acc, metrics, all_targets, all_preds, all_probs

    def fit(self):
        """Entrenar el modelo"""

        print(f"\n🚀 Iniciando entrenamiento de {self.model_name}")
        print("=" * 60)

        # Setup
        set_seed(42)
        self.setup_model()
        self.load_data()

        # Early stopping
        early_stopping = EarlyStopping(
            patience=self.config['patience'],
            restore_best_weights=True
        )

        # Variables de control
        best_val_acc = 0.0
        start_time = time.time()

        # Loop de entrenamiento
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['epochs']}")
            print("-" * 40)

            # Entrenar
            train_loss, train_acc = self.train_epoch()

            # Validar
            val_loss, val_acc, val_metrics, _, _, _ = self.validate_epoch()

            # Actualizar scheduler
            if self.config['scheduler'] == 'cosine':
                self.scheduler.step()
            elif self.config['scheduler'] == 'plateau':
                self.scheduler.step(val_acc)

            # Guardar historia
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)

            # Imprimir estadísticas
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f} | LR: {current_lr:.2e}")

            # Guardar mejor modelo
            if val_acc > best_val_acc:
                best_val_acc = val_acc

                model_path = f"results/models/best_{self.model_name.lower()}.pth"
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss, val_acc, model_path
                )
                print(f"🎯 Nuevo mejor modelo: {val_acc:.4f}")

            # Early stopping
            if early_stopping(val_acc, self.model):
                print(f"⏹️  Early stopping en época {epoch + 1}")
                break

        # Tiempo total
        total_time = time.time() - start_time
        print(f"\n⏱️  Tiempo total: {total_time:.1f}s")
        print(f"🏆 Mejor Val Accuracy: {best_val_acc:.4f}")

        return self.history

    def evaluate(self, data_loader, dataset_name="Test"):
        """Evaluar modelo en un dataset"""

        print(f"\n📊 Evaluando en {dataset_name}")
        print("-" * 40)

        self.model.eval()

        all_preds = []
        all_targets = []
        all_probs = []

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(data_loader, desc=f'Evaluating {dataset_name}'):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                predicted = output.argmax(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                # Guardar para métricas
                probs = F.softmax(output, dim=1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Calcular métricas
        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total
        metrics = calculate_metrics(all_targets, all_preds, np.array(all_probs))

        # Imprimir resultados
        print(f"{dataset_name} Results:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        if 'auc' in metrics:
            print(f"  AUC: {metrics['auc']:.4f}")

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'metrics': metrics,
            'predictions': all_preds,
            'targets': all_targets,
            'probabilities': all_probs
        }

    def plot_training_history(self, save_path=None):
        """Graficar historia de entrenamiento"""

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        epochs = range(1, len(self.history['train_loss']) + 1)

        # Loss
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        axes[0].set_title('Loss durante el Entrenamiento')
        axes[0].set_xlabel('Épocas')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Accuracy
        axes[1].plot(epochs, self.history['train_acc'], 'b-', label='Train Acc')
        axes[1].plot(epochs, self.history['val_acc'], 'r-', label='Val Acc')
        axes[1].set_title('Accuracy durante el Entrenamiento')
        axes[1].set_xlabel('Épocas')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        # Learning Rate
        axes[2].plot(epochs, self.history['learning_rates'], 'g-')
        axes[2].set_title('Learning Rate')
        axes[2].set_xlabel('Épocas')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_yscale('log')
        axes[2].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Gráfico guardado: {save_path}")

        plt.show()

        return fig

    def plot_confusion_matrix(self, targets, predictions, save_path=None):
        """Graficar matriz de confusión"""

        cm = confusion_matrix(targets, predictions)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Sin Grieta', 'Con Grieta'],
                    yticklabels=['Sin Grieta', 'Con Grieta'])

        plt.title(f'Matriz de Confusión - {self.model_name}')
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')

        # Agregar estadísticas
        accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
        plt.text(0.5, -0.1, f'Accuracy: {accuracy:.4f}',
                 ha='center', transform=plt.gca().transAxes)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Matriz de confusión guardada: {save_path}")

        plt.show()

        return cm

    def save_results(self, test_results):
        """Guardar resultados del experimento"""

        results = {
            'model_name': self.model_name,
            'config': self.config,
            'history': self.history,
            'test_results': {
                'loss': test_results['loss'],
                'accuracy': test_results['accuracy'],
                'metrics': test_results['metrics']
            },
            'best_val_accuracy': max(self.history['val_acc']),
            'total_epochs': len(self.history['train_loss'])
        }

        # Guardar como JSON
        results_path = f"results/logs/{self.model_name.lower()}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✅ Resultados guardados: {results_path}")

        return results


def train_model(model_name, data_dir='data/raw', config=None):
    """Función principal para entrenar un modelo"""

    trainer = CrackTrainer(model_name, data_dir, config)

    # Entrenar
    history = trainer.fit()

    # Evaluar en test
    test_results = trainer.evaluate(trainer.test_loader, "Test")

    # Visualizaciones
    trainer.plot_training_history(f"results/plots/{model_name.lower()}_training.png")
    trainer.plot_confusion_matrix(
        test_results['targets'],
        test_results['predictions'],
        f"results/plots/{model_name.lower()}_confusion.png"
    )

    # Guardar resultados
    results = trainer.save_results(test_results)

    return trainer, results


if __name__ == "__main__":
    """Entrenar modelo individual"""

    import argparse

    parser = argparse.ArgumentParser(description='Entrenar modelo de detección de grietas')
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['simplecnn', 'resnet18', 'efficientnet_b0'],
                        help='Modelo a entrenar')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Número de épocas')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Tamaño de batch')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')

    args = parser.parse_args()

    # Configuración personalizada
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
    }

    print(f"🚀 Entrenando {args.model} con configuración personalizada")

    # Entrenar
    trainer, results = train_model(args.model, config=config)

    print(f"\n🎉 Entrenamiento completado!")
    print(f"🏆 Test Accuracy: {results['test_results']['accuracy']:.4f}")
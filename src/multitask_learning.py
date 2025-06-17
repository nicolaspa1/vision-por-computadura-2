#!/usr/bin/env python3
"""
Extensión 2: Multi-Task Learning para Grietas
Objetivo: Detectar grietas + Clasificar severidad + Localizar región
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import get_model
import numpy as np


class MultiTaskCrackModel(nn.Module):
    """Modelo multi-task para detección de grietas con múltiples salidas"""

    def __init__(self, backbone_name='resnet18'):
        super().__init__()

        # Backbone compartido
        if backbone_name == 'resnet18':
            import torchvision.models as models
            backbone = models.resnet18(pretrained=True)
            self.features = nn.Sequential(*list(backbone.children())[:-2])  # Sin avgpool y fc
            feature_dim = 512

        # Pooling adaptativo
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Cabezas de tarea específicas

        # 1. Detección binaria (crack/no-crack)
        self.detection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Binary classification
        )

        # 2. Clasificación de severidad (4 niveles)
        self.severity_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # None, Light, Moderate, Severe
        )

        # 3. Localización de región (4 cuadrantes)
        self.location_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # Top-left, Top-right, Bottom-left, Bottom-right
        )

        # 4. Regresión de área afectada (0-100%)
        self.area_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output 0-1
        )

    def forward(self, x):
        # Características compartidas
        features = self.features(x)  # [batch, 512, H, W]

        # Múltiples salidas
        detection = self.detection_head(features)  # [batch, 2]
        severity = self.severity_head(features)  # [batch, 4]
        location = self.location_head(features)  # [batch, 4]
        area = self.area_head(features)  # [batch, 1]

        return {
            'detection': detection,
            'severity': severity,
            'location': location,
            'area': area
        }


class MultiTaskTrainer:
    """Entrenador para modelo multi-task"""

    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device

        # Criterios de pérdida para cada tarea
        self.detection_criterion = nn.CrossEntropyLoss()
        self.severity_criterion = nn.CrossEntropyLoss()
        self.location_criterion = nn.CrossEntropyLoss()
        self.area_criterion = nn.MSELoss()

        # Pesos para balancear las pérdidas
        self.task_weights = {
            'detection': 1.0,
            'severity': 0.5,
            'location': 0.3,
            'area': 0.2
        }

    def compute_multi_task_loss(self, outputs, targets):
        """Calcular pérdida combinada para todas las tareas"""

        losses = {}

        # 1. Pérdida de detección
        losses['detection'] = self.detection_criterion(
            outputs['detection'], targets['detection']
        )

        # 2. Pérdida de severidad (solo para imágenes con grietas)
        if 'severity' in targets:
            losses['severity'] = self.severity_criterion(
                outputs['severity'], targets['severity']
            )
        else:
            losses['severity'] = torch.tensor(0.0, device=self.device)

        # 3. Pérdida de localización (solo para imágenes con grietas)
        if 'location' in targets:
            losses['location'] = self.location_criterion(
                outputs['location'], targets['location']
            )
        else:
            losses['location'] = torch.tensor(0.0, device=self.device)

        # 4. Pérdida de área
        if 'area' in targets:
            losses['area'] = self.area_criterion(
                outputs['area'].squeeze(), targets['area'].float()
            )
        else:
            losses['area'] = torch.tensor(0.0, device=self.device)

        # Pérdida total ponderada
        total_loss = sum(
            self.task_weights[task] * loss
            for task, loss in losses.items()
        )

        losses['total'] = total_loss
        return losses


def create_synthetic_multilabels(data_loader):
    """Crear etiquetas sintéticas para multi-task learning"""

    multi_task_data = []

    for images, labels in data_loader:
        batch_size = images.shape[0]

        # Crear etiquetas multi-task sintéticas
        multi_labels = {
            'detection': labels,  # Original crack/no-crack
            'severity': torch.zeros(batch_size, dtype=torch.long),
            'location': torch.zeros(batch_size, dtype=torch.long),
            'area': torch.zeros(batch_size)
        }

        # Para imágenes con grietas, generar etiquetas sintéticas
        for i in range(batch_size):
            if labels[i] == 1:  # Tiene grieta
                # Severidad aleatoria basada en simulación
                multi_labels['severity'][i] = torch.randint(1, 4, (1,))  # Light, Moderate, Severe

                # Localización aleatoria
                multi_labels['location'][i] = torch.randint(0, 4, (1,))  # 4 cuadrantes

                # Área afectada (0.1 - 0.8 para grietas)
                multi_labels['area'][i] = torch.rand(1) * 0.7 + 0.1
            else:
                # Sin grieta
                multi_labels['severity'][i] = 0  # None
                multi_labels['location'][i] = 0  # N/A
                multi_labels['area'][i] = 0.0  # 0% area

        multi_task_data.append((images, multi_labels))

    return multi_task_data


def compare_single_vs_multitask():
    """Comparar enfoque single-task vs multi-task"""

    print("🔬 COMPARACIÓN: Single-Task vs Multi-Task Learning")
    print("=" * 60)

    # Cargar datos
    from utils import get_data_loaders
    train_loader, val_loader, test_loader = get_data_loaders('data/raw', batch_size=32)

    # Crear datos multi-task sintéticos
    multi_train_data = create_synthetic_multilabels(train_loader)

    print("✅ Datos multi-task creados")
    print(f"📊 Tareas: Detección + Severidad + Localización + Área")

    # Modelo multi-task
    multi_model = MultiTaskCrackModel('resnet18')
    trainer = MultiTaskTrainer(multi_model)

    print("✅ Modelo multi-task configurado")
    print(f"🎯 Salidas: {multi_model(torch.randn(1, 3, 224, 224)).keys()}")

    # Comparación teórica
    comparison = {
        'single_task': {
            'objective': 'Solo detección crack/no-crack',
            'outputs': 1,
            'complexity': 'Baja',
            'accuracy': '99.85% (actual)',
            'applications': 'Detección simple'
        },
        'multi_task': {
            'objective': 'Detección + Severidad + Localización + Área',
            'outputs': 4,
            'complexity': 'Alta',
            'accuracy': 'Estimado 97-99% (más difícil)',
            'applications': 'Análisis completo de grietas'
        }
    }

    print("\n📊 COMPARACIÓN DETALLADA:")
    print("-" * 60)
    for approach, details in comparison.items():
        print(f"\n{approach.upper()}:")
        for key, value in details.items():
            print(f"  {key}: {value}")

    return multi_model, trainer


def main():
    """Función principal del análisis multi-task"""

    print("🎯 ANÁLISIS MULTI-TASK LEARNING")
    print("=" * 60)

    # Comparar enfoques
    multi_model, trainer = compare_single_vs_multitask()

    # Demostración de arquitectura
    dummy_input = torch.randn(2, 3, 224, 224)
    outputs = multi_model(dummy_input)

    print(f"\n🔍 DEMOSTRACIÓN DE SALIDAS:")
    print("-" * 40)
    for task, output in outputs.items():
        print(f"{task}: {output.shape}")

    print(f"\n💡 INSIGHTS MULTI-TASK:")
    print("✅ Aprovecha características compartidas")
    print("✅ Una sola red para múltiples análisis")
    print("✅ Más información por inferencia")
    print("⚠️ Mayor complejidad de entrenamiento")
    print("⚠️ Requiere etiquetas más ricas")

    return multi_model


if __name__ == "__main__":
    main()
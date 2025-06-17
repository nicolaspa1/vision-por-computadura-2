#!/usr/bin/env python3
"""
Arquitecturas de modelos para detección de grietas en concreto
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class SimpleCNN(nn.Module):
    """CNN básica como modelo baseline"""

    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # Primer bloque convolucional
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 227 -> 113

            # Segundo bloque convolucional
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 113 -> 56

            # Tercer bloque convolucional
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56 -> 28

            # Cuarto bloque convolucional
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # -> 1x1
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResNetBaseline(nn.Module):
    """ResNet18 con transfer learning"""

    def __init__(self, num_classes=2, pretrained=True):
        super(ResNetBaseline, self).__init__()

        # Cargar ResNet18 preentrenado
        self.backbone = timm.create_model('resnet18', pretrained=pretrained, num_classes=0)

        # Congelar capas iniciales (opcional)
        self.freeze_backbone = False
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Clasificador personalizado
        feature_dim = self.backbone.num_features  # 512 para ResNet18
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class AttentionModule(nn.Module):
    """Módulo de atención simple"""

    def __init__(self, feature_dim):
        super(AttentionModule, self).__init__()

        self.spatial_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dim, feature_dim // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 8, feature_dim, 1),
            nn.Sigmoid()
        )

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Atención espacial
        spatial_att = self.spatial_attention(x)
        x = x * spatial_att

        # Atención de canal
        b, c, h, w = x.size()
        channel_att = self.channel_attention(x).view(b, c, 1, 1)
        x = x * channel_att

        return x


class EfficientNetWithAttention(nn.Module):
    """EfficientNet con mecanismo de atención"""

    def __init__(self, num_classes=2, model_name='efficientnet_b0', pretrained=True):
        super(EfficientNetWithAttention, self).__init__()

        # Backbone EfficientNet
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True
        )

        # Obtener número de características de la última capa
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dim = features[-1].shape[1]

        # Módulo de atención
        self.attention = AttentionModule(feature_dim)

        # Pooling adaptativo
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Clasificador
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Extraer características
        features = self.backbone(x)
        x = features[-1]  # Usar la última capa de características

        # Aplicar atención
        x = self.attention(x)

        # Pooling global
        x = self.global_pool(x)

        # Clasificación
        x = self.classifier(x)

        return x


class CrackDetectorEnsemble(nn.Module):
    """Ensemble de múltiples modelos"""

    def __init__(self, models, weights=None):
        super(CrackDetectorEnsemble, self).__init__()

        self.models = nn.ModuleList(models)

        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = weights

    def forward(self, x):
        outputs = []

        for model in self.models:
            with torch.no_grad():
                output = model(x)
                outputs.append(F.softmax(output, dim=1))

        # Promedio ponderado
        ensemble_output = torch.zeros_like(outputs[0])
        for output, weight in zip(outputs, self.weights):
            ensemble_output += weight * output

        return ensemble_output


def get_model(model_name, num_classes=2, pretrained=True):
    """Factory function para crear modelos"""

    if model_name.lower() == 'simplecnn':
        return SimpleCNN(num_classes=num_classes)

    elif model_name.lower() == 'resnet18':
        return ResNetBaseline(num_classes=num_classes, pretrained=pretrained)

    elif model_name.lower() in ['efficientnet', 'efficientnet_b0']:
        return EfficientNetWithAttention(
            num_classes=num_classes,
            model_name='efficientnet_b0',
            pretrained=pretrained
        )

    elif model_name.lower() == 'efficientnet_b1':
        return EfficientNetWithAttention(
            num_classes=num_classes,
            model_name='efficientnet_b1',
            pretrained=pretrained
        )

    elif model_name.lower() == 'efficientnet_b2':
        return EfficientNetWithAttention(
            num_classes=num_classes,
            model_name='efficientnet_b2',
            pretrained=pretrained
        )

    else:
        raise ValueError(f"Modelo no reconocido: {model_name}")


def count_parameters(model):
    """Contar parámetros del modelo"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def model_summary(model, input_size=(3, 224, 224)):
    """Resumen del modelo"""

    params = count_parameters(model)

    print(f"Modelo: {model.__class__.__name__}")
    print(f"Parámetros totales: {params['total']:,}")
    print(f"Parámetros entrenables: {params['trainable']:,}")
    print(f"Parámetros congelados: {params['non_trainable']:,}")

    # Test forward pass
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, *input_size)
        try:
            output = model(dummy_input)
            print(f"Tamaño de salida: {output.shape}")
        except Exception as e:
            print(f"Error en forward pass: {e}")


if __name__ == "__main__":
    """Test de los modelos"""

    print("🧠 Probando arquitecturas de modelos...")
    print("=" * 60)

    models_to_test = [
        ('SimpleCNN', 'simplecnn'),
        ('ResNet18', 'resnet18'),
        ('EfficientNet-B0', 'efficientnet_b0'),
    ]

    for model_name, model_key in models_to_test:
        print(f"\n🔍 {model_name}")
        print("-" * 30)

        try:
            model = get_model(model_key, num_classes=2, pretrained=True)
            model_summary(model)
        except Exception as e:
            print(f"❌ Error creando {model_name}: {e}")

    print("\n✅ Test de modelos completado!")
#!/usr/bin/env python3
"""
Análisis exploratorio rápido del dataset de grietas
Ejecutar: python src/exploratory_analysis.py
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
from pathlib import Path
import random


def analyze_dataset():
    """Análisis básico del dataset"""

    data_dir = Path("../data/raw")
    positive_dir = data_dir / "Positive"
    negative_dir = data_dir / "Negative"

    if not positive_dir.exists() or not negative_dir.exists():
        print("❌ Dataset no encontrado. Ejecuta primero: python download_data.py")
        return False

    # Contar imágenes
    positive_images = list(positive_dir.glob("*.jpg")) + list(positive_dir.glob("*.png"))
    negative_images = list(negative_dir.glob("*.jpg")) + list(negative_dir.glob("*.png"))

    print("📊 ANÁLISIS DEL DATASET")
    print("=" * 50)
    print(f"🔴 Imágenes con grietas: {len(positive_images)}")
    print(f"🟢 Imágenes sin grietas: {len(negative_images)}")
    print(f"📈 Total: {len(positive_images) + len(negative_images)}")
    print(f"⚖️  Balance: {len(positive_images) / (len(positive_images) + len(negative_images)) * 100:.1f}% positivas")

    return positive_images, negative_images


def analyze_image_properties(positive_images, negative_images):
    """Analizar propiedades de las imágenes"""

    print("\n🔍 ANÁLISIS DE PROPIEDADES")
    print("=" * 50)

    # Muestrear algunas imágenes
    sample_positive = random.sample(positive_images, min(10, len(positive_images)))
    sample_negative = random.sample(negative_images, min(10, len(negative_images)))

    widths, heights = [], []

    for img_path in sample_positive + sample_negative:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
        except Exception as e:
            print(f"⚠️  Error leyendo {img_path}: {e}")

    if widths and heights:
        print(f"📐 Dimensiones:")
        print(f"   Ancho: {min(widths)} - {max(widths)} (promedio: {np.mean(widths):.0f})")
        print(f"   Alto: {min(heights)} - {max(heights)} (promedio: {np.mean(heights):.0f})")

        # Verificar si todas tienen el mismo tamaño
        if len(set(widths)) == 1 and len(set(heights)) == 1:
            print(f"✅ Todas las imágenes tienen el mismo tamaño: {widths[0]}x{heights[0]}")
        else:
            print("⚠️  Las imágenes tienen diferentes tamaños - necesitaremos redimensionar")


def create_sample_visualization(positive_images, negative_images):
    """Crear visualización de muestras"""

    print("\n🎨 Creando visualización de muestras...")

    # Configurar matplotlib
    plt.style.use('default')
    fig, axes = plt.subplots(2, 8, figsize=(16, 6))
    fig.suptitle('Muestras del Dataset de Grietas en Concreto', fontsize=16, fontweight='bold')

    # Seleccionar muestras aleatorias
    sample_positive = random.sample(positive_images, min(8, len(positive_images)))
    sample_negative = random.sample(negative_images, min(8, len(negative_images)))

    # Mostrar imágenes positivas (con grietas)
    for i, img_path in enumerate(sample_positive):
        try:
            img = Image.open(img_path)
            axes[0, i].imshow(img)
            axes[0, i].set_title('Con Grieta', fontsize=10, color='red')
            axes[0, i].axis('off')
        except Exception as e:
            axes[0, i].text(0.5, 0.5, 'Error', ha='center', va='center', transform=axes[0, i].transAxes)
            axes[0, i].axis('off')

    # Mostrar imágenes negativas (sin grietas)
    for i, img_path in enumerate(sample_negative):
        try:
            img = Image.open(img_path)
            axes[1, i].imshow(img)
            axes[1, i].set_title('Sin Grieta', fontsize=10, color='green')
            axes[1, i].axis('off')
        except Exception as e:
            axes[1, i].text(0.5, 0.5, 'Error', ha='center', va='center', transform=axes[1, i].transAxes)
            axes[1, i].axis('off')

    # Limpiar axes vacíos
    for i in range(len(sample_positive), 8):
        axes[0, i].axis('off')
    for i in range(len(sample_negative), 8):
        axes[1, i].axis('off')

    plt.tight_layout()

    # Guardar figura
    os.makedirs('../results/plots', exist_ok=True)
    plt.savefig('results/plots/dataset_samples.png', dpi=300, bbox_inches='tight')
    print("✅ Visualización guardada: results/plots/dataset_samples.png")

    plt.show()


def create_distribution_plot(positive_images, negative_images):
    """Crear gráfico de distribución de clases"""

    plt.figure(figsize=(10, 6))

    # Datos para el gráfico
    classes = ['Sin Grieta', 'Con Grieta']
    counts = [len(negative_images), len(positive_images)]
    colors = ['green', 'red']

    # Gráfico de barras
    plt.subplot(1, 2, 1)
    bars = plt.bar(classes, counts, color=colors, alpha=0.7, edgecolor='black')
    plt.title('Distribución de Clases', fontsize=14, fontweight='bold')
    plt.ylabel('Número de Imágenes')

    # Agregar números en las barras
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                 str(count), ha='center', va='bottom', fontweight='bold')

    # Gráfico de pie
    plt.subplot(1, 2, 2)
    plt.pie(counts, labels=classes, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Proporción de Clases', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/plots/class_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Distribución guardada: results/plots/class_distribution.png")

    plt.show()


def generate_summary_report(positive_images, negative_images):
    """Generar reporte resumen"""

    total_images = len(positive_images) + len(negative_images)
    balance_ratio = len(positive_images) / total_images * 100

    report = f"""
# 📊 Reporte de Análisis Exploratorio

## Dataset: Detección de Grietas en Concreto

### 📈 Estadísticas Básicas
- **Total de imágenes**: {total_images:,}
- **Imágenes con grietas**: {len(positive_images):,} ({balance_ratio:.1f}%)
- **Imágenes sin grietas**: {len(negative_images):,} ({100 - balance_ratio:.1f}%)

### 🎯 Balance de Clases
{'✅ Dataset balanceado' if 40 <= balance_ratio <= 60 else '⚠️ Dataset desbalanceado'}

### 📁 Archivos Generados
- `results/plots/dataset_samples.png` - Muestras del dataset
- `results/plots/class_distribution.png` - Distribución de clases

---
*Generado en exploratory_analysis.py*
"""

    # Guardar reporte
    with open('../results/exploratory_report.md', 'w', encoding='utf-8') as f:
        f.write(report)

    print("\n📄 Reporte generado: results/exploratory_report.md")
    print(report)


def main():
    """Función principal del análisis exploratorio"""

    print("🔍 ANÁLISIS EXPLORATORIO DE DATOS")
    print("=" * 60)

    # Analizar dataset
    result = analyze_dataset()
    if not result:
        return False

    positive_images, negative_images = result

    # Analizar propiedades
    analyze_image_properties(positive_images, negative_images)

    # Crear visualizaciones
    create_sample_visualization(positive_images, negative_images)
    create_distribution_plot(positive_images, negative_images)

    # Generar reporte
    generate_summary_report(positive_images, negative_images)

    print("\n¡Análisis exploratorio completado!")
    print("\nSiguiente paso: python run_experiment.py")

    return True


if __name__ == "__main__":
    main()
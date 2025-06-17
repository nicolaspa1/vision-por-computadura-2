#!/usr/bin/env python3
"""
Script para descargar el dataset de grietas en concreto usando kagglehub
Ejecutar: python download_data.py
"""

import os
import shutil
from pathlib import Path


def download_dataset():
    """Extraer dataset de grietas desde archivo local archive.zip"""

    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    print("📥 Configurando dataset de grietas en concreto...")

    # Verificar si ya existe
    if (data_dir / "Positive").exists() and (data_dir / "Negative").exists():
        print("✅ Dataset ya existe!")
        return True

    # Buscar archivo archive.zip en el directorio actual
    zip_files = [
        Path("archive.zip"),
        Path("../archive.zip"),
        Path("../../archive.zip"),
        data_dir / "archive.zip"
    ]

    zip_path = None
    for potential_zip in zip_files:
        if potential_zip.exists():
            zip_path = potential_zip
            break

    if zip_path:
        print(f"📦 Archivo encontrado: {zip_path}")
        try:
            import zipfile

            print("📂 Extrayendo archivos...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)

            print("✅ Extracción completada")

            # Verificar estructura
            verify_dataset_structure(data_dir)

            return True

        except Exception as e:
            print(f"❌ Error extrayendo archivo: {e}")
            return False

    else:
        print("❌ Archivo archive.zip no encontrado")
        print("\n📋 Instrucciones:")
        print("1. Asegúrate de que archive.zip esté en una de estas ubicaciones:")
        for zip_file in zip_files:
            print(f"   - {zip_file}")
        print("2. O copia archive.zip al directorio del proyecto")
        print("3. Ejecuta nuevamente: python download_data.py")

        # Ofrecer crear dataset de muestra
        print("\n🤔 ¿Quieres crear un dataset de muestra para empezar? (y/n)")
        try:
            response = input().lower().strip()
            if response in ['y', 'yes', 'sí', 's']:
                return create_sample_dataset()
        except:
            pass

        return False


def verify_dataset_structure(data_dir):
    """Verificar que la estructura del dataset es correcta"""

    positive_dir = data_dir / "Positive"
    negative_dir = data_dir / "Negative"

    if not positive_dir.exists():
        # Buscar carpetas alternativas
        for folder in data_dir.iterdir():
            if folder.is_dir() and "positive" in folder.name.lower():
                folder.rename(positive_dir)
                break

    if not negative_dir.exists():
        # Buscar carpetas alternativas
        for folder in data_dir.iterdir():
            if folder.is_dir() and "negative" in folder.name.lower():
                folder.rename(negative_dir)
                break

    # Contar imágenes
    if positive_dir.exists() and negative_dir.exists():
        pos_count = len(list(positive_dir.glob("*.jpg"))) + len(list(positive_dir.glob("*.png")))
        neg_count = len(list(negative_dir.glob("*.jpg"))) + len(list(negative_dir.glob("*.png")))

        print(f"✅ Dataset verificado:")
        print(f"   📷 Imágenes positivas (con grietas): {pos_count}")
        print(f"   📷 Imágenes negativas (sin grietas): {neg_count}")
        print(f"   📊 Total: {pos_count + neg_count}")

        if pos_count > 0 and neg_count > 0:
            return True

    print("❌ Estructura del dataset incorrecta")
    return False


def create_sample_dataset():
    """Crear un dataset de muestra para pruebas rápidas"""

    print("🧪 Creando dataset de muestra para pruebas...")

    try:
        import torch
        import torchvision.transforms as transforms
        from PIL import Image, ImageDraw
        import random

        sample_dir = Path("data/raw")
        positive_dir = sample_dir / "Positive"
        negative_dir = sample_dir / "Negative"

        positive_dir.mkdir(parents=True, exist_ok=True)
        negative_dir.mkdir(parents=True, exist_ok=True)

        # Crear imágenes sintéticas simples
        for i in range(50):  # 50 por clase para pruebas
            # Imagen base
            img = Image.new('RGB', (224, 224), color=(128, 128, 128))
            draw = ImageDraw.Draw(img)

            # Agregar textura de concreto (ruido)
            for _ in range(1000):
                x, y = random.randint(0, 223), random.randint(0, 223)
                color = random.randint(100, 150)
                draw.point((x, y), fill=(color, color, color))

            # Imagen positiva (con grieta)
            if i < 25:
                # Dibujar grieta
                start_x = random.randint(0, 224)
                start_y = random.randint(0, 224)
                end_x = random.randint(0, 224)
                end_y = random.randint(0, 224)
                draw.line([(start_x, start_y), (end_x, end_y)], fill=(50, 50, 50), width=2)

                img.save(positive_dir / f"crack_{i:04d}.jpg")

            # Imagen negativa (sin grieta)
            else:
                img.save(negative_dir / f"no_crack_{i:04d}.jpg")

        print("✅ Dataset de muestra creado (100 imágenes)")
        return True

    except Exception as e:
        print(f"❌ Error creando dataset de muestra: {e}")
        return False


def main():
    """Función principal"""

    print("🔍 Configurando dataset para detección de grietas...")
    print("=" * 60)

    # Intentar descarga automática
    if not download_dataset():
        print("\n🤔 ¿Quieres crear un dataset de muestra para empezar? (y/n)")
        response = input().lower().strip()

        if response in ['y', 'yes', 'sí', 's']:
            create_sample_dataset()
        else:
            print("⏸️  Configuración pausada. Descarga el dataset manualmente.")
            return False

    print("\n🎉 ¡Dataset listo para usar!")
    print("\n📂 Próximos pasos:")
    print("1. Ejecutar: python src/exploratory_analysis.py")
    print("2. Entrenar modelos: python run_experiment.py")

    return True


if __name__ == "__main__":
    main()
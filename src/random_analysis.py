#!/usr/bin/env python3
"""
Análisis aleatorio de imágenes usando el mejor modelo entrenado
Clasifica imágenes al azar del dataset y muestra predicciones vs realidad
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from pathlib import Path
import json
from torchvision import transforms
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import time

# Importar nuestros módulos
import sys

sys.path.append('')
from models import get_model
from utils import get_device


class RandomCrackAnalyzer:
    """Analizador aleatorio de grietas usando el mejor modelo"""

    def __init__(self, model_path=None, data_dir="data/raw"):
        self.device = get_device()
        self.data_dir = Path(data_dir)

        # Cargar el mejor modelo
        self.model = self.load_best_model(model_path)

        # Configurar transformaciones (mismo que entrenamiento)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Cargar rutas de imágenes
        self.image_paths = self.load_image_paths()

        print(f"🎯 Analizador listo!")
        print(f"📊 Total de imágenes disponibles: {len(self.image_paths)}")
        print(f"🧠 Modelo: {self.model.__class__.__name__}")
        print(f"💻 Device: {self.device}")

    def load_best_model(self, model_path=None):
        """Cargar el mejor modelo entrenado"""

        # Si no se especifica path, buscar el mejor modelo
        if model_path is None:
            possible_paths = [
                "results/models/best_resnet18_optimized.pth",
                "results/models/best_resnet18.pth",
                "results/models/best_simplecnn.pth"
            ]

            for path in possible_paths:
                if Path(path).exists():
                    model_path = path
                    break

            if model_path is None:
                raise FileNotFoundError("❌ No se encontró ningún modelo entrenado")

        print(f"📥 Cargando modelo desde: {model_path}")

        # Determinar arquitectura del modelo por el nombre del archivo
        if "resnet18" in str(model_path).lower():
            model = get_model("resnet18", num_classes=2, pretrained=False)
        elif "simplecnn" in str(model_path).lower():
            model = get_model("simplecnn", num_classes=2, pretrained=False)
        else:
            # Default a ResNet18
            model = get_model("resnet18", num_classes=2, pretrained=False)

        # Cargar pesos
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            # Si es un checkpoint completo
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Modelo cargado (época {checkpoint.get('epoch', 'N/A')})")
                print(f"   Accuracy: {checkpoint.get('accuracy', 'N/A'):.4f}")
            else:
                # Si es solo el state_dict
                model.load_state_dict(checkpoint)
                print("✅ Modelo cargado (solo pesos)")

        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            # Fallback: crear modelo nuevo para demo
            print("🔄 Creando modelo nuevo para demostración...")
            model = get_model("resnet18", num_classes=2, pretrained=True)

        model = model.to(self.device)
        model.eval()

        return model

    def load_image_paths(self):
        """Cargar rutas de todas las imágenes disponibles"""

        image_paths = []

        # Imágenes positivas (con grietas)
        positive_dir = self.data_dir / "Positive"
        if positive_dir.exists():
            for img_path in positive_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    image_paths.append((img_path, 1, "Con Grieta"))

        # Imágenes negativas (sin grietas)
        negative_dir = self.data_dir / "Negative"
        if negative_dir.exists():
            for img_path in negative_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    image_paths.append((img_path, 0, "Sin Grieta"))

        return image_paths

    def predict_single_image(self, image_path):
        """Predecir una sola imagen"""

        start_time = time.time()

        # Cargar y preprocessar imagen
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Inferencia
            with torch.no_grad():
                logits = self.model(image_tensor)
                probabilities = F.softmax(logits, dim=1)
                predicted_class = logits.argmax(dim=1).item()
                confidence = probabilities.max().item()

            inference_time = time.time() - start_time

            return {
                'predicted_class': predicted_class,
                'predicted_label': "Con Grieta" if predicted_class == 1 else "Sin Grieta",
                'confidence': confidence,
                'probabilities': probabilities.cpu().numpy()[0],
                'inference_time': inference_time,
                'image': image
            }

        except Exception as e:
            print(f"❌ Error procesando {image_path}: {e}")
            return None

    def analyze_random_samples(self, n_samples=12, save_results=True):
        """Analizar muestras aleatorias del dataset"""

        print(f"\n🎲 Analizando {n_samples} imágenes aleatorias...")
        print("=" * 60)

        # Seleccionar muestras aleatorias
        random_samples = random.sample(self.image_paths, n_samples)

        # Analizar cada muestra
        results = []
        correct_predictions = 0

        for i, (image_path, true_label, true_label_text) in enumerate(random_samples, 1):
            print(f"🔍 Analizando {i}/{n_samples}: {image_path.name}")

            # Predicción
            prediction = self.predict_single_image(image_path)

            if prediction is None:
                continue

            # Verificar si es correcta
            is_correct = prediction['predicted_class'] == true_label
            if is_correct:
                correct_predictions += 1

            # Guardar resultado
            result = {
                'image_path': image_path,
                'true_label': true_label,
                'true_label_text': true_label_text,
                'predicted_class': prediction['predicted_class'],
                'predicted_label': prediction['predicted_label'],
                'confidence': prediction['confidence'],
                'probabilities': prediction['probabilities'],
                'is_correct': is_correct,
                'inference_time': prediction['inference_time'],
                'image': prediction['image']
            }

            results.append(result)

            # Mostrar resultado en tiempo real
            status = "✅" if is_correct else "❌"
            print(
                f"   {status} Real: {true_label_text} | Predicho: {prediction['predicted_label']} | Confianza: {prediction['confidence']:.2%}")

        # Estadísticas generales
        accuracy = correct_predictions / len(results) if results else 0
        avg_confidence = np.mean([r['confidence'] for r in results]) if results else 0
        avg_time = np.mean([r['inference_time'] for r in results]) if results else 0

        print(f"\n📊 RESULTADOS DEL ANÁLISIS ALEATORIO:")
        print(f"   🎯 Accuracy: {accuracy:.2%} ({correct_predictions}/{len(results)})")
        print(f"   🔮 Confianza promedio: {avg_confidence:.2%}")
        print(f"   ⚡ Tiempo promedio: {avg_time * 1000:.1f}ms por imagen")

        # Crear visualización
        self.visualize_random_analysis(results)

        # Guardar resultados si se solicita
        if save_results:
            self.save_analysis_results(results, accuracy, avg_confidence, avg_time)

        return results

    def visualize_random_analysis(self, results):
        """Crear visualización de los resultados"""

        n_samples = len(results)
        cols = 4
        rows = (n_samples + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        if rows == 1:
            axes = [axes]
        if cols == 1:
            axes = [[ax] for ax in axes]

        for i, result in enumerate(results):
            row, col = i // cols, i % cols
            ax = axes[row][col] if rows > 1 else axes[col]

            # Mostrar imagen
            ax.imshow(result['image'])

            # Título con información
            is_correct = result['is_correct']
            color = 'green' if is_correct else 'red'
            status = "CORRECTO" if is_correct else "ERROR"

            title = f"{status}\n"
            title += f"Real: {result['true_label_text']}\n"
            title += f"Predicho: {result['predicted_label']}\n"
            title += f"Confianza: {result['confidence']:.1%}"

            ax.set_title(title, fontsize=10, color=color, fontweight='bold')
            ax.axis('off')

        # Ocultar axes vacíos
        for i in range(len(results), rows * cols):
            row, col = i // cols, i % cols
            ax = axes[row][col] if rows > 1 else axes[col]
            ax.axis('off')

        plt.tight_layout()
        plt.savefig('results/plots/random_analysis_grid.png', dpi=300, bbox_inches='tight')
        print("✅ Visualización guardada: results/plots/random_analysis_grid.png")
        plt.show()

        # Crear gráfico de distribución de confianza
        self.plot_confidence_distribution(results)

    def plot_confidence_distribution(self, results):
        """Graficar distribución de confianza"""

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Separar por correctas/incorrectas
        correct_conf = [r['confidence'] for r in results if r['is_correct']]
        incorrect_conf = [r['confidence'] for r in results if not r['is_correct']]

        # Histograma de confianza
        axes[0].hist(correct_conf, alpha=0.7, label='Correctas', color='green', bins=10)
        axes[0].hist(incorrect_conf, alpha=0.7, label='Incorrectas', color='red', bins=10)
        axes[0].set_xlabel('Confianza del Modelo')
        axes[0].set_ylabel('Frecuencia')
        axes[0].set_title('Distribución de Confianza')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Matriz de confusión
        true_labels = [r['true_label'] for r in results]
        pred_labels = [r['predicted_class'] for r in results]

        cm = confusion_matrix(true_labels, pred_labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                    xticklabels=['Sin Grieta', 'Con Grieta'],
                    yticklabels=['Sin Grieta', 'Con Grieta'])
        axes[1].set_title('Matriz de Confusión')
        axes[1].set_xlabel('Predicción')
        axes[1].set_ylabel('Real')

        plt.tight_layout()
        plt.savefig('results/plots/random_analysis_stats.png', dpi=300, bbox_inches='tight')
        print("✅ Estadísticas guardadas: results/plots/random_analysis_stats.png")
        plt.show()

    def save_analysis_results(self, results, accuracy, avg_confidence, avg_time):
        """Guardar resultados del análisis"""

        # Preparar datos para JSON (sin imágenes)
        json_results = []
        for result in results:
            json_result = {
                'image_name': result['image_path'].name,
                'true_label': result['true_label'],
                'true_label_text': result['true_label_text'],
                'predicted_class': result['predicted_class'],
                'predicted_label': result['predicted_label'],
                'confidence': float(result['confidence']),
                'probabilities': result['probabilities'].tolist(),
                'is_correct': result['is_correct'],
                'inference_time': float(result['inference_time'])
            }
            json_results.append(json_result)

        # Resumen
        summary = {
            'total_samples': len(results),
            'correct_predictions': sum(1 for r in results if r['is_correct']),
            'accuracy': float(accuracy),
            'average_confidence': float(avg_confidence),
            'average_inference_time': float(avg_time),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        # Guardar
        analysis_data = {
            'summary': summary,
            'results': json_results
        }

        output_path = 'results/logs/random_analysis_results.json'
        with open(output_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)

        print(f"✅ Resultados guardados: {output_path}")

    def interactive_analysis(self):
        """Modo interactivo para análisis continuo"""

        print("\n🎮 MODO INTERACTIVO ACTIVADO")
        print("=" * 50)
        print("Presiona Enter para analizar una imagen aleatoria")
        print("Escribe 'q' para salir")
        print("Escribe un número para analizar N imágenes")

        while True:
            try:
                user_input = input("\n➤ ").strip().lower()

                if user_input == 'q' or user_input == 'quit':
                    print("👋 ¡Hasta luego!")
                    break
                elif user_input == '':
                    # Analizar una imagen
                    self.analyze_random_samples(n_samples=1, save_results=False)
                elif user_input.isdigit():
                    # Analizar N imágenes
                    n = int(user_input)
                    if 1 <= n <= 50:
                        self.analyze_random_samples(n_samples=n, save_results=True)
                    else:
                        print("⚠️ Por favor ingresa un número entre 1 y 50")
                else:
                    print("⚠️ Comando no reconocido. Presiona Enter, escribe un número, o 'q' para salir")

            except KeyboardInterrupt:
                print("\n👋 ¡Hasta luego!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Función principal"""

    print("🔍 ANÁLISIS ALEATORIO DE DETECCIÓN DE GRIETAS")
    print("=" * 60)
    print("🎯 Usando el mejor modelo entrenado para clasificar imágenes al azar")

    try:
        # Crear analizador
        analyzer = RandomCrackAnalyzer()

        # Menú de opciones
        print(f"\n📋 OPCIONES DISPONIBLES:")
        print("1. Análisis rápido (12 imágenes)")
        print("2. Análisis extensivo (25 imágenes)")
        print("3. Modo interactivo")
        print("4. Análisis personalizado")

        choice = input("\nSelecciona una opción (1-4): ").strip()

        if choice == '1':
            analyzer.analyze_random_samples(n_samples=12)
        elif choice == '2':
            analyzer.analyze_random_samples(n_samples=25)
        elif choice == '3':
            analyzer.interactive_analysis()
        elif choice == '4':
            try:
                n = int(input("¿Cuántas imágenes quieres analizar? (1-100): "))
                if 1 <= n <= 100:
                    analyzer.analyze_random_samples(n_samples=n)
                else:
                    print("⚠️ Número fuera de rango")
            except ValueError:
                print("⚠️ Por favor ingresa un número válido")
        else:
            print("⚠️ Opción no válida")
            return

        print(f"\n🎉 Análisis completado!")
        print(f"📁 Revisa los resultados en results/plots/ y results/logs/")

    except Exception as e:
        print(f"❌ Error en el análisis: {e}")
        print("💡 Asegúrate de que:")
        print("   - El dataset esté en data/raw/")
        print("   - Exista al menos un modelo entrenado en results/models/")


if __name__ == "__main__":
    main()
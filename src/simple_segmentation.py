#!/usr/bin/env python3
"""
Análisis de Segmentación de Grietas - Versión Simplificada
Solo usa OpenCV para evitar dependencias complejas
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random


class SimpleCrackSegmentationAnalyzer:
    """Analizador de segmentación usando solo OpenCV"""

    def __init__(self):
        self.methods = {
            'canny': self.canny_segmentation,
            'adaptive': self.adaptive_segmentation,
            'morphological': self.morphological_segmentation,
            'otsu': self.otsu_segmentation
        }

    def canny_segmentation(self, gray_image):
        """Segmentación usando Canny Edge Detection"""
        return cv2.Canny(gray_image, 50, 150, apertureSize=3)

    def adaptive_segmentation(self, gray_image):
        """Segmentación usando Adaptive Threshold"""
        return cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)

    def morphological_segmentation(self, gray_image):
        """Segmentación usando operaciones morfológicas"""
        # Threshold adaptativo primero
        adaptive = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)

        # Operaciones morfológicas para limpiar
        kernel = np.ones((3, 3), np.uint8)
        return cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)

    def otsu_segmentation(self, gray_image):
        """Segmentación usando Otsu Threshold"""
        _, result = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return result

    def analyze_single_image(self, image_path):
        """Analizar una sola imagen con todos los métodos"""

        # Cargar imagen
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"❌ No se pudo cargar: {image_path}")
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Aplicar todos los métodos
        results = {}
        for method_name, method_func in self.methods.items():
            try:
                results[method_name] = method_func(gray)
            except Exception as e:
                print(f"⚠️ Error en {method_name}: {e}")
                results[method_name] = np.zeros_like(gray)

        # Calcular métricas
        metrics = self.calculate_metrics(results, gray.shape)

        return {
            'original': img,
            'gray': gray,
            'segmentations': results,
            'metrics': metrics,
            'image_path': image_path
        }

    def calculate_metrics(self, segmentations, image_shape):
        """Calcular métricas"""

        total_pixels = image_shape[0] * image_shape[1]
        metrics = {}

        for method_name, segmented in segmentations.items():
            # Pixeles detectados como grieta
            crack_pixels = np.sum(segmented > 0)
            crack_percentage = (crack_pixels / total_pixels) * 100

            # Número de componentes conectados
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(segmented)

            # Filtrar componentes pequeños (ruido)
            min_area = 20
            valid_components = stats[1:, cv2.CC_STAT_AREA] > min_area
            num_significant_components = np.sum(valid_components)

            metrics[method_name] = {
                'crack_percentage': crack_percentage,
                'num_components': num_significant_components,
                'total_crack_pixels': crack_pixels
            }

        return metrics

    def create_comparison_visualization(self, analysis_result):
        """Crear visualización comparativa de métodos"""

        if analysis_result is None:
            return None

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Imagen original
        original_rgb = cv2.cvtColor(analysis_result['original'], cv2.COLOR_BGR2RGB)
        axes[0, 0].imshow(original_rgb)
        axes[0, 0].set_title('Imagen Original', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')

        # Imagen en escala de grises
        axes[0, 1].imshow(analysis_result['gray'], cmap='gray')
        axes[0, 1].set_title('Escala de Grises', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')

        # Métodos de segmentación
        methods_info = [
            ('canny', 'Canny Edge Detection'),
            ('adaptive', 'Adaptive Threshold'),
            ('morphological', 'Morphological Operations'),
            ('otsu', 'Otsu Threshold')
        ]

        positions = [(0, 2), (1, 0), (1, 1), (1, 2)]

        for (method, title), (row, col) in zip(methods_info, positions):
            if method in analysis_result['segmentations']:
                segmented = analysis_result['segmentations'][method]
                metrics = analysis_result['metrics'][method]

                axes[row, col].imshow(segmented, cmap='hot')

                # Título con métricas
                full_title = f"{title}\n{metrics['crack_percentage']:.1f}% área detectada"
                axes[row, col].set_title(full_title, fontsize=12, fontweight='bold')
                axes[row, col].axis('off')

        plt.tight_layout()
        return fig

    def analyze_dataset_sample(self, data_dir, n_samples=8):
        """Analizar muestra del dataset"""

        positive_dir = Path(data_dir) / "Positive"
        negative_dir = Path(data_dir) / "Negative"

        if not positive_dir.exists() or not negative_dir.exists():
            print(f"❌ Directorios no encontrados: {data_dir}")
            return []

        # Seleccionar muestras
        positive_images = list(positive_dir.glob("*.jpg"))[:n_samples // 2]
        negative_images = list(negative_dir.glob("*.jpg"))[:n_samples // 2]

        all_images = positive_images + negative_images

        if not all_images:
            print("❌ No se encontraron imágenes")
            return []

        print(f"🔍 Analizando {len(all_images)} imágenes...")

        results = []
        for i, img_path in enumerate(all_images):
            print(f"Procesando {i + 1}/{len(all_images)}: {img_path.name}")

            # Analizar imagen
            analysis = self.analyze_single_image(img_path)
            if analysis:
                results.append(analysis)

                # Crear y guardar visualización
                fig = self.create_comparison_visualization(analysis)
                if fig:
                    output_path = f"results/plots/segmentation_{i}_{img_path.stem}.png"
                    fig.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    print(f"  ✅ Guardado: {output_path}")

        return results

    def generate_summary_report(self, results):
        """Generar reporte resumen del análisis"""

        if not results:
            print("❌ No hay resultados para el reporte")
            return

        # Separar por tipo de imagen
        positive_results = [r for r in results if 'Positive' in str(r['image_path'])]
        negative_results = [r for r in results if 'Negative' in str(r['image_path'])]

        # Estadísticas
        method_stats = {}

        for method in self.methods.keys():
            pos_percentages = [r['metrics'][method]['crack_percentage'] for r in positive_results]
            neg_percentages = [r['metrics'][method]['crack_percentage'] for r in negative_results]

            method_stats[method] = {
                'positive_avg': np.mean(pos_percentages) if pos_percentages else 0,
                'negative_avg': np.mean(neg_percentages) if neg_percentages else 0,
                'positive_std': np.std(pos_percentages) if pos_percentages else 0,
                'negative_std': np.std(neg_percentages) if neg_percentages else 0
            }

        # Crear gráfico resumen
        self.create_summary_plot(method_stats)

        # Generar reporte escrito
        report = f"""
# 🔍 Análisis de Segmentación de Grietas

## 📊 Resumen de Resultados

### Imágenes Analizadas
- **Total**: {len(results)} imágenes
- **Con grietas**: {len(positive_results)} imágenes  
- **Sin grietas**: {len(negative_results)} imágenes

### Estadísticas por Método

"""

        for method, stats in method_stats.items():
            report += f"""
#### {method.upper()}
- **Imágenes con grietas**: {stats['positive_avg']:.1f}% ± {stats['positive_std']:.1f}%
- **Imágenes sin grietas**: {stats['negative_avg']:.1f}% ± {stats['negative_std']:.1f}%
"""

        report += f"""

## 💡 Conclusiones

1. **Detección de grietas**: Los métodos clásicos identifican patrones de grietas
2. **Variabilidad**: Diferentes métodos detectan diferentes aspectos
3. **Complementariedad**: Combinar métodos puede mejorar robustez
4. **Aplicación práctica**: Base para pipeline híbrido con deep learning

## 📁 Archivos Generados
- Análisis individual: `results/plots/segmentation_*.png`
- Resumen estadístico: `results/plots/segmentation_summary.png`

---
"""

        # Guardar reporte
        with open('results/segmentation_report.md', 'w', encoding='utf-8') as f:
            f.write(report)

        print("✅ Reporte guardado: results/segmentation_report.md")
        print(report)

        return report

    def create_summary_plot(self, method_stats):
        """Crear gráfico resumen de estadísticas"""

        methods = list(method_stats.keys())
        positive_avgs = [method_stats[m]['positive_avg'] for m in methods]
        negative_avgs = [method_stats[m]['negative_avg'] for m in methods]

        x = np.arange(len(methods))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))

        bars1 = ax.bar(x - width / 2, positive_avgs, width, label='Imágenes con grietas',
                       color='red', alpha=0.7)
        bars2 = ax.bar(x + width / 2, negative_avgs, width, label='Imágenes sin grietas',
                       color='green', alpha=0.7)

        ax.set_xlabel('Métodos de Segmentación')
        ax.set_ylabel('Porcentaje de Área Detectada (%)')
        ax.set_title('Comparación de Métodos de Segmentación')
        ax.set_xticks(x)
        ax.set_xticklabels([m.title() for m in methods])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Agregar valores en las barras
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                        f'{height:.1f}%', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('results/plots/segmentation_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ Gráfico resumen: results/plots/segmentation_summary.png")


def main():
    """Función principal"""

    print("🔍 ANÁLISIS SIMPLIFICADO DE SEGMENTACIÓN")
    print("=" * 60)

    # Crear analizador
    analyzer = SimpleCrackSegmentationAnalyzer()

    # Analizar muestra del dataset
    results = analyzer.analyze_dataset_sample("data/raw", n_samples=8)

    if results:
        # Generar reporte
        analyzer.generate_summary_report(results)

        print(f"\n🎉 Análisis completado!")
        print(f"📊 {len(results)} imágenes analizadas")
        print(f"📁 Resultados en results/plots/")
    else:
        print("❌ No se pudieron analizar las imágenes")


if __name__ == "__main__":
    main()
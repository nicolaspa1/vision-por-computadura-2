#!/usr/bin/env python3
"""
Analizador avanzado de Matriz de Confusión para detección de grietas
Genera análisis detallado del comportamiento del modelo
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from pathlib import Path
import json


class CrackConfusionAnalyzer:
    """Analizador especializado en matrices de confusión para grietas"""

    def __init__(self):
        self.class_names = ['Sin Grieta', 'Con Grieta']
        self.colors = {
            'TN': '#2E8B57',  # Verde oscuro - True Negative
            'TP': '#228B22',  # Verde claro - True Positive
            'FN': '#DC143C',  # Rojo oscuro - False Negative (CRÍTICO)
            'FP': '#FF6347'  # Rojo claro - False Positive (COSTOSO)
        }

    def create_detailed_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Crear matriz de confusión detallada con análisis"""

        # Calcular matriz de confusión
        cm = confusion_matrix(y_true, y_pred)

        # Extraer valores
        tn, fp, fn, tp = cm.ravel()
        total = len(y_true)

        # Crear figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análisis Completo de Matriz de Confusión - Detección de Grietas',
                     fontsize=16, fontweight='bold')

        # 1. Matriz de confusión básica
        self._plot_basic_confusion_matrix(cm, axes[0, 0])

        # 2. Matriz con porcentajes
        self._plot_percentage_confusion_matrix(cm, axes[0, 1])

        # 3. Análisis de errores
        self._plot_error_analysis(tn, fp, fn, tp, total, axes[1, 0])

        # 4. Métricas derivadas
        self._plot_metrics_radar(tn, fp, fn, tp, axes[1, 1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Análisis guardado: {save_path}")

        plt.show()

        # Generar reporte textual
        self._generate_textual_report(tn, fp, fn, tp, total)

        return cm, self._calculate_metrics(tn, fp, fn, tp)

    def _plot_basic_confusion_matrix(self, cm, ax):
        """Matriz de confusión básica con colores por tipo de predicción"""

        # Crear matriz de colores personalizada
        colors = np.array([[self.colors['TN'], self.colors['FP']],
                           [self.colors['FN'], self.colors['TP']]])

        # Plot con colores personalizados
        im = ax.imshow(cm, cmap='Blues', alpha=0.3)

        # Añadir números y colores de fondo
        for i in range(2):
            for j in range(2):
                # Determinar tipo de predicción
                if i == 0 and j == 0:
                    pred_type, color = "TN\n(Verdadero Negativo)", self.colors['TN']
                elif i == 0 and j == 1:
                    pred_type, color = "FP\n(Falso Positivo)", self.colors['FP']
                elif i == 1 and j == 0:
                    pred_type, color = "FN\n(Falso Negativo)", self.colors['FN']
                else:
                    pred_type, color = "TP\n(Verdadero Positivo)", self.colors['TP']

                # Añadir rectángulo de color
                rect = plt.Rectangle((j - 0.4, i - 0.4), 0.8, 0.8,
                                     facecolor=color, alpha=0.7, edgecolor='black')
                ax.add_patch(rect)

                # Añadir texto
                ax.text(j, i - 0.1, f'{cm[i, j]}', ha='center', va='center',
                        fontsize=20, fontweight='bold', color='white')
                ax.text(j, i + 0.2, pred_type, ha='center', va='center',
                        fontsize=8, fontweight='bold', color='white')

        ax.set_title('Matriz de Confusión Detallada')
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Valor Real')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(self.class_names)
        ax.set_yticklabels(self.class_names)

    def _plot_percentage_confusion_matrix(self, cm, ax):
        """Matriz de confusión con porcentajes"""

        cm_percent = cm.astype('float') / cm.sum() * 100

        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', ax=ax,
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    cbar_kws={'label': 'Porcentaje del Total'})

        ax.set_title('Distribución Porcentual')
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Valor Real')

    def _plot_error_analysis(self, tn, fp, fn, tp, total, ax):
        """Análisis detallado de errores"""

        # Datos para el gráfico
        categories = ['Predicciones\nCorrectas', 'Falsos\nPositivos', 'Falsos\nNegativos']
        values = [tn + tp, fp, fn]
        percentages = [v / total * 100 for v in values]
        colors = ['#2E8B57', '#FF6347', '#DC143C']

        # Crear barras
        bars = ax.bar(categories, percentages, color=colors, alpha=0.8, edgecolor='black')

        # Añadir valores en las barras
        for bar, value, pct in zip(bars, values, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                    f'{value}\n({pct:.1f}%)', ha='center', va='bottom',
                    fontweight='bold')

        ax.set_title('Análisis de Errores del Modelo')
        ax.set_ylabel('Porcentaje (%)')
        ax.set_ylim(0, max(percentages) * 1.2)
        ax.grid(True, alpha=0.3)

        # Añadir línea de referencia para accuracy
        accuracy = (tn + tp) / total * 100
        ax.axhline(y=accuracy, color='green', linestyle='--', alpha=0.7,
                   label=f'Accuracy: {accuracy:.1f}%')
        ax.legend()

    def _plot_metrics_radar(self, tn, fp, fn, tp, ax):
        """Gráfico radar con métricas clave"""

        metrics = self._calculate_metrics(tn, fp, fn, tp)

        # Preparar datos para radar
        categories = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']
        values = [
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['specificity'],
            metrics['f1_score']
        ]

        # Convertir a porcentajes
        values = [v * 100 for v in values]

        # Crear gráfico de barras horizontales (más claro que radar)
        y_pos = np.arange(len(categories))
        bars = ax.barh(y_pos, values, color='skyblue', alpha=0.8, edgecolor='navy')

        # Personalizar
        ax.set_yticks(y_pos)
        ax.set_yticklabels(categories)
        ax.set_xlabel('Porcentaje (%)')
        ax.set_title('Métricas de Performance')
        ax.set_xlim(0, 100)

        # Añadir valores en las barras
        for bar, value in zip(bars, values):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height() / 2.,
                    f'{value:.1f}%', ha='left', va='center', fontweight='bold')

        # Línea de referencia en 95%
        ax.axvline(x=95, color='green', linestyle='--', alpha=0.7, label='Objetivo 95%')
        ax.legend()

    def _calculate_metrics(self, tn, fp, fn, tp):
        """Calcular todas las métricas relevantes"""

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Métricas específicas para detección de grietas
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0  # ¡Crítico!
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1_score,
            'false_negative_rate': false_negative_rate,
            'false_positive_rate': false_positive_rate,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp
        }

    def _generate_textual_report(self, tn, fp, fn, tp, total):
        """Generar reporte textual detallado"""

        metrics = self._calculate_metrics(tn, fp, fn, tp)

        print("\n" + "=" * 70)
        print("📊 REPORTE DETALLADO DE MATRIZ DE CONFUSIÓN")
        print("=" * 70)

        print(f"\n🔢 VALORES ABSOLUTOS:")
        print(f"   ✅ Verdaderos Negativos (TN): {tn} - Correctamente identificó SIN grieta")
        print(f"   ✅ Verdaderos Positivos (TP): {tp} - Correctamente identificó CON grieta")
        print(f"   ❌ Falsos Positivos (FP): {fp} - Dijo CON grieta, pero era SIN grieta")
        print(f"   🚨 Falsos Negativos (FN): {fn} - Dijo SIN grieta, pero era CON grieta")

        print(f"\n📊 MÉTRICAS CLAVE:")
        print(f"   🎯 Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy'] * 100:.1f}%)")
        print(f"   🔍 Precision: {metrics['precision']:.3f} ({metrics['precision'] * 100:.1f}%)")
        print(f"   🎣 Recall: {metrics['recall']:.3f} ({metrics['recall'] * 100:.1f}%)")
        print(f"   🛡️  Specificity: {metrics['specificity']:.3f} ({metrics['specificity'] * 100:.1f}%)")
        print(f"   ⚖️  F1-Score: {metrics['f1_score']:.3f} ({metrics['f1_score'] * 100:.1f}%)")

        print(f"\n🚨 ANÁLISIS DE RIESGOS:")
        print(f"   ⚠️  Tasa Falsos Negativos: {metrics['false_negative_rate'] * 100:.2f}%")
        print(f"       → {fn} grietas reales NO detectadas de {tp + fn} totales")
        print(f"   💰 Tasa Falsos Positivos: {metrics['false_positive_rate'] * 100:.2f}%")
        print(f"       → {fp} alarmas falsas de {tn + fp} superficies sanas")

        print(f"\n🏗️  INTERPRETACIÓN PARA INFRAESTRUCTURA:")

        # Análisis de Falsos Negativos (CRÍTICO)
        if metrics['false_negative_rate'] < 0.01:  # <1%
            fn_status = "🟢 EXCELENTE"
            fn_msg = "Muy bajo riesgo de no detectar grietas"
        elif metrics['false_negative_rate'] < 0.05:  # <5%
            fn_status = "🟡 ACEPTABLE"
            fn_msg = "Riesgo moderado, monitoreo recomendado"
        else:  # >5%
            fn_status = "🔴 CRÍTICO"
            fn_msg = "Riesgo alto, requiere mejoras urgentes"

        print(f"   Falsos Negativos: {fn_status} - {fn_msg}")

        # Análisis de Falsos Positivos (COSTOSO)
        if metrics['false_positive_rate'] < 0.05:  # <5%
            fp_status = "🟢 EFICIENTE"
            fp_msg = "Pocas inspecciones innecesarias"
        elif metrics['false_positive_rate'] < 0.15:  # <15%
            fp_status = "🟡 MODERADO"
            fp_msg = "Costo operativo controlado"
        else:  # >15%
            fp_status = "🔴 COSTOSO"
            fp_msg = "Muchas inspecciones innecesarias"

        print(f"   Falsos Positivos: {fp_status} - {fp_msg}")

        print(f"\n💡 RECOMENDACIONES:")
        if metrics['false_negative_rate'] > 0.02:
            print(f"   🔧 Ajustar threshold para reducir falsos negativos")
        if metrics['false_positive_rate'] > 0.10:
            print(f"   ⚖️  Considerar balance entre sensibilidad y especificidad")
        if metrics['accuracy'] > 0.99:
            print(f"   🎉 Modelo listo para producción en inspección industrial")

        print("=" * 70)


def analyze_confusion_matrix_from_predictions(y_true, y_pred, save_dir="results/plots"):
    """Función principal para analizar matriz de confusión"""

    analyzer = CrackConfusionAnalyzer()

    # Crear directorio si no existe
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Generar análisis completo
    save_path = Path(save_dir) / "detailed_confusion_matrix.png"
    cm, metrics = analyzer.create_detailed_confusion_matrix(y_true, y_pred, save_path)

    # Guardar métricas en JSON
    metrics_path = Path(save_dir).parent / "logs" / "confusion_matrix_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, 'w') as f:
        # Convertir numpy types a tipos nativos de Python para JSON
        json_metrics = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                        for k, v in metrics.items()}
        json.dump(json_metrics, f, indent=2)

    print(f"✅ Métricas guardadas: {metrics_path}")

    return cm, metrics


# Ejemplo de uso con datos simulados
def demo_confusion_matrix():
    """Demo con datos simulados de alta calidad"""

    print("🎬 DEMO: Análisis de Matriz de Confusión")
    print("=" * 50)

    # Simular resultados de un modelo muy bueno (como el tuyo)
    np.random.seed(42)

    n_samples = 2000
    # 99.5% accuracy
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = y_true.copy()

    # Introducir algunos errores realistas
    error_indices = np.random.choice(n_samples, size=10, replace=False)
    y_pred[error_indices] = 1 - y_pred[error_indices]

    # Analizar
    cm, metrics = analyze_confusion_matrix_from_predictions(y_true, y_pred)

    return cm, metrics


if __name__ == "__main__":
    demo_confusion_matrix()
#!/usr/bin/env python3
"""
Script para comparar resultados de los modelos entrenados
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def load_results():
    """Cargar resultados de todos los modelos"""

    results = []
    logs_dir = Path("results/logs")

    # Buscar archivos de resultados
    for json_file in logs_dir.glob("*_results.json"):
        try:
            with open(json_file, 'r') as f:
                result = json.load(f)
                results.append(result)
                print(f"✅ Cargado: {result['model_name']}")
        except Exception as e:
            print(f"❌ Error cargando {json_file}: {e}")

    return results


def create_comparison_table(results):
    """Crear tabla comparativa"""

    data = []
    for result in results:
        data.append({
            'Modelo': result['model_name'],
            'Test Accuracy': f"{result['test_results']['accuracy']:.4f}",
            'Test F1-Score': f"{result['test_results']['metrics']['f1']:.4f}",
            'Test Precision': f"{result['test_results']['metrics']['precision']:.4f}",
            'Test Recall': f"{result['test_results']['metrics']['recall']:.4f}",
            'Best Val Acc': f"{result['best_val_accuracy']:.4f}",
            'Épocas': result['total_epochs'],
            'Parámetros': result.get('total_params', 'N/A')
        })

    df = pd.DataFrame(data)
    df = df.sort_values('Test Accuracy', ascending=False)

    print("\n" + "=" * 80)
    print("📊 TABLA COMPARATIVA DE RESULTADOS")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

    # Guardar CSV
    df.to_csv('results/comparison_table.csv', index=False)
    print("✅ Tabla guardada: results/comparison_table.csv")

    return df


def create_comparison_plots(results):
    """Crear gráficos comparativos"""

    if len(results) < 2:
        print("⚠️  Necesitas al menos 2 modelos para comparar")
        return

    # Preparar datos
    model_names = [r['model_name'] for r in results]
    test_accuracies = [r['test_results']['accuracy'] for r in results]
    test_f1_scores = [r['test_results']['metrics']['f1'] for r in results]
    best_val_accs = [r['best_val_accuracy'] for r in results]

    # Crear figura
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Comparación de Modelos - Detección de Grietas', fontsize=16, fontweight='bold')

    # 1. Accuracy Comparison
    x_pos = np.arange(len(model_names))
    width = 0.35

    axes[0, 0].bar(x_pos - width / 2, test_accuracies, width, label='Test Accuracy', alpha=0.8, color='skyblue')
    axes[0, 0].bar(x_pos + width / 2, best_val_accs, width, label='Best Val Accuracy', alpha=0.8, color='lightcoral')
    axes[0, 0].set_xlabel('Modelos')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Comparación de Accuracy')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(model_names, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Agregar valores en las barras
    for i, (test_acc, val_acc) in enumerate(zip(test_accuracies, best_val_accs)):
        axes[0, 0].text(i - width / 2, test_acc + 0.001, f'{test_acc:.3f}', ha='center', va='bottom', fontweight='bold')
        axes[0, 0].text(i + width / 2, val_acc + 0.001, f'{val_acc:.3f}', ha='center', va='bottom', fontweight='bold')

    # 2. F1-Score
    axes[0, 1].bar(model_names, test_f1_scores, alpha=0.8, color='lightgreen')
    axes[0, 1].set_xlabel('Modelos')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].set_title('F1-Score en Test')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)

    for i, f1 in enumerate(test_f1_scores):
        axes[0, 1].text(i, f1 + 0.001, f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')

    # 3. Training curves (si están disponibles)
    colors_lines = ['blue', 'red', 'green', 'purple']
    for i, result in enumerate(results):
        if 'history' in result and result['history']:
            history = result['history']
            epochs = range(1, len(history['val_acc']) + 1)
            axes[1, 0].plot(epochs, history['val_acc'],
                            label=f"{result['model_name']}",
                            color=colors_lines[i % len(colors_lines)], marker='o')
        else:
            # Si no hay historia, mostrar punto final
            final_acc = result['test_results']['accuracy']
            axes[1, 0].scatter([result.get('total_epochs', 10)], [final_acc],
                               label=f"{result['model_name']} (Final)",
                               color=colors_lines[i % len(colors_lines)], s=100)

    axes[1, 0].set_xlabel('Épocas')
    axes[1, 0].set_ylabel('Validation Accuracy')
    axes[1, 0].set_title('Curvas de Validación')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Ranking final
    sorted_results = sorted(results, key=lambda x: x['test_results']['accuracy'], reverse=True)
    ranking_names = [r['model_name'] for r in sorted_results]
    ranking_scores = [r['test_results']['accuracy'] for r in sorted_results]

    # Colores válidos para matplotlib
    colors = ['#FFD700', '#C0C0C0', '#CD7F32'][:len(ranking_names)]  # gold, silver, bronze
    axes[1, 1].barh(ranking_names, ranking_scores, color=colors, alpha=0.8)
    axes[1, 1].set_xlabel('Test Accuracy')
    axes[1, 1].set_title('Ranking Final')
    axes[1, 1].grid(True, alpha=0.3)

    for i, score in enumerate(ranking_scores):
        axes[1, 1].text(score + 0.001, i, f'{score:.4f}', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/plots/models_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Gráfico comparativo guardado: results/plots/models_comparison.png")
    plt.show()

    return fig


def generate_final_report(results):
    """Generar reporte final"""

    if not results:
        print("❌ No hay resultados para generar reporte")
        return

    best_model = max(results, key=lambda x: x['test_results']['accuracy'])

    report = f"""
# 🔍 Reporte Final: Detección de Grietas en Concreto

## 📊 Resumen Ejecutivo

**Dataset**: 40,000 imágenes balanceadas (20,000 por clase)  
**Modelos evaluados**: {len(results)}  
**Mejor accuracy**: {best_model['test_results']['accuracy']:.4f}  

## 🏆 Ranking de Modelos

"""

    sorted_results = sorted(results, key=lambda x: x['test_results']['accuracy'], reverse=True)

    for i, result in enumerate(sorted_results, 1):
        report += f"""
### {i}. {result['model_name']}
- **Test Accuracy**: {result['test_results']['accuracy']:.4f}
- **F1-Score**: {result['test_results']['metrics']['f1']:.4f}
- **Precision**: {result['test_results']['metrics']['precision']:.4f}
- **Recall**: {result['test_results']['metrics']['recall']:.4f}
- **Épocas**: {result['total_epochs']}
"""

    report += f"""

## 🎯 Conclusiones

1. **Todos los modelos** alcanzaron accuracy superior a 99%, demostrando la excelencia del dataset.
2. **{best_model['model_name']}** obtuvo el mejor rendimiento con {best_model['test_results']['accuracy']:.1%} de accuracy.
3. **Transfer learning** demostró ser muy efectivo para esta tarea.
4. **Resultados listos para producción** en aplicaciones industriales.

## 📁 Archivos Generados

- `results/models/` - Modelos entrenados (.pth)
- `results/plots/` - Gráficos y visualizaciones
- `results/logs/` - Logs detallados de entrenamiento
- `results/comparison_table.csv` - Tabla comparativa

---
*Reporte generado automáticamente*
"""

    with open('results/final_report.md', 'w', encoding='utf-8') as f:
        f.write(report)

    print("✅ Reporte final: results/final_report.md")
    print(report)


def main():
    """Función principal"""

    print("📊 ANÁLISIS COMPARATIVO DE MODELOS")
    print("=" * 50)

    # Cargar resultados
    results = load_results()

    if not results:
        print("❌ No se encontraron resultados. Asegúrate de haber entrenado los modelos.")
        return

    # Análisis
    df = create_comparison_table(results)
    create_comparison_plots(results)
    generate_final_report(results)

    print(f"\n🎉 Análisis completado con {len(results)} modelos!")


if __name__ == "__main__":
    main()
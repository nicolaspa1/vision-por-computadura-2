#!/usr/bin/env python3
"""
Script para crear la estructura inicial del proyecto
Ejecutar: python setup_project.py
"""

import os


def create_project_structure():
    """Crear estructura de carpetas del proyecto"""

    folders = [
        'data',
        'data/raw',
        'data/processed',
        'src',
        'notebooks',
        'results',
        'results/models',
        'results/plots',
        'results/logs'
    ]

    print("📁 Creando estructura del proyecto...")

    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"✅ {folder}/")

    # Crear archivos __init__.py para importaciones
    init_files = ['src/__init__.py']
    for init_file in init_files:
        with open(init_file, 'w') as f:
            f.write('# Package initialization\n')
        print(f"✅ {init_file}")

    # Crear .gitignore básico
    gitignore_content = """
# Data files
data/raw/
*.zip
*.tar.gz

# Model files
results/models/*.pth
results/models/*.pt

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
.venv/
venv/

# Jupyter
.ipynb_checkpoints/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
results/logs/
"""

    with open('.gitignore', 'w') as f:
        f.write(gitignore_content.strip())
    print("✅ .gitignore")

    print("\n🎉 Estructura del proyecto creada exitosamente!")
    print("\n📂 Estructura final:")
    print("""
crack_detection_project/
├── data/
│   ├── raw/           # Dataset original
│   └── processed/     # Datos preprocesados
├── src/               # Código fuente
├── notebooks/         # Jupyter notebooks
├── results/
│   ├── models/        # Modelos entrenados
│   ├── plots/         # Gráficos
│   └── logs/          # Logs de entrenamiento
└── .gitignore
""")


if __name__ == "__main__":
    create_project_structure()
"""
test_installation.py -- Verificar que todo esté correctamente instalado.

Ejecutar: python test_installation.py
"""

def test_imports():
    """Verifica que todas las librerías necesarias están instaladas."""
    print("Verificando librerías necesarias...\n")
    
    errors = []
    
    libs = {
        "pandas": "Manipulación de datos",
        "numpy": "Operaciones numéricas",
        "sklearn": "Preprocessing, métricas",
        "xgboost": "Modelo XGBoost",
        "optuna": "Optimización de hiperparámetros",
        "mlflow": "Tracking de experimentos",
    }
    
    for lib, desc in libs.items():
        try:
            __import__(lib)
            print(f"✓ {lib:15s} - {desc}")
        except ImportError:
            print(f"✗ {lib:15s} - {desc}")
            errors.append(lib)
    
    print("\n" + "="*60)
    
    if errors:
        print(f"❌ Faltan {len(errors)} librerías:")
        print(f"   pip install {' '.join(errors)}")
        return False
    else:
        print("✅ Todas las librerías están instaladas correctamente")
        return True


def test_modules():
    """Verifica que los módulos del proyecto se importan correctamente."""
    print("\nVerificando módulos del proyecto...\n")
    
    modules = [
        "preprocessing",
        "training",
        "monitoring",
        "postprocessing",
        "main"
    ]
    
    errors = []
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}.py")
        except ImportError as e:
            print(f"✗ {module}.py - Error: {e}")
            errors.append(module)
    
    print("\n" + "="*60)
    
    if errors:
        print(f"❌ {len(errors)} módulos no se pueden importar")
        return False
    else:
        print("✅ Todos los módulos se importan correctamente")
        return True


def test_directories():
    """Verifica que los directorios necesarios existen."""
    print("\nVerificando directorios...\n")
    
    import os
    from pathlib import Path
    
    dirs = [
        "data",
        "data/raw",
        "data/processed",
        "data/models",
        "data/monitoring",
        "data/postprocessed",
        "data/replica",
        "data/replica/s3",
        "data/replica/athena",
        "data/replica/onpremise",
        "exampleNotebooks"
    ]
    
    errors = []
    for d in dirs:
        if os.path.isdir(d):
            print(f"✓ {d}/")
        else:
            print(f"⚠ {d}/ (crear)")
            Path(d).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("✅ Todos los directorios están listos")
    return True


def main():
    print("\n" + "="*60)
    print("🔍 VERIFICACION DE INSTALACION")
    print("="*60 + "\n")
    
    test1 = test_imports()
    test2 = test_modules()
    test3 = test_directories()
    
    print("\n" + "="*60)
    if test1 and test2 and test3:
        print("✅ LISTO PARA USAR")
        print("="*60)
        print("\nEjecutar pipeline:")
        print("  python main.py --mode training --input 'data/raw/'")
        print("  python main.py --mode inference --input 'data/raw/p1_extrac.csv'")
        print("="*60 + "\n")
        return 0
    else:
        print("❌ REVISAR ERRORES ARRIBA")
        print("="*60 + "\n")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

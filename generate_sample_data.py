"""
generate_sample_data.py
Generates synthetic sample data for testing the ML pipeline.
No heavy dependencies required - only pandas and numpy.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_sample_data(n_samples=1000, random_seed=42):
    """
    Generate synthetic dataset matching the expected schema:
    - p_codmes: Code month
    - key_value: Unique identifier
    - target: Binary target (0/1) 
    - grp_campecs06m: Campaign segment (categorical)
    - monto: Transaction amount
    - prob_value_contact: Contact probability (0-1)
    - prob_frescura: Freshness probability (0-1)
    - seg_un: Segmentation (categorical)
    - riesgo: Risk level (categorical)
    - edad: Age (numeric)
    - ingreso_xxx: Income features (numeric)
    """
    np.random.seed(random_seed)
    
    today = datetime.now()
    n_months = 12
    dates = [today - timedelta(days=30*i) for i in range(n_months)]
    
    data = {
        'p_codmes': np.random.choice([d.strftime('%Y%m') for d in dates], n_samples),
        'key_value': np.arange(n_samples),
        'target': np.random.binomial(1, 0.3, n_samples),  # 30% positive class
        'grp_campecs06m': np.random.choice(['G1', 'G2', 'G3', 'G4', 'G5'], n_samples),
        'monto': np.random.exponential(scale=1000, size=n_samples),
        'prob_value_contact': np.random.uniform(0, 1, n_samples),
        'prob_frescura': np.random.uniform(0, 1, n_samples),
        'seg_un': np.random.choice(['SEG_ALTO', 'SEG_MEDIO', 'SEG_BAJO'], n_samples),
        'riesgo': np.random.choice(['BAJO', 'MEDIO', 'ALTO', 'MUY_ALTO'], n_samples),
        'edad': np.random.normal(loc=45, scale=15, size=n_samples).astype(int),
        'ingreso_xxx_1': np.random.exponential(scale=50000, size=n_samples),
        'ingreso_xxx_2': np.random.exponential(scale=30000, size=n_samples),
        'ingreso_xxx_3': np.random.exponential(scale=20000, size=n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Add some NaN values to simulate real data
    for col in ['ingreso_xxx_1', 'ingreso_xxx_2', 'ingreso_xxx_3']:
        missing_idx = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
        df.loc[missing_idx, col] = np.nan
    
    return df

def main():
    """Generate and save sample data files."""
    print("=" * 60)
    print("🔧 Generador de Datos de Prueba - ML Pipeline")
    print("=" * 60)
    print()
    
    # Create data directories
    dirs = ['data/raw', 'data/processed', 'data/models', 'data/monitoring', 
            'data/postprocessed', 'data/replica/s3', 'data/replica/athena', 'data/replica/onpremise']
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print("✓ Directorios creados")
    print()
    
    # Generate 10 sample files (p1_extrac.csv through p10_extrac.csv)
    print("📊 Generando 10 archivos de datos...")
    for i in range(1, 11):
        df = generate_sample_data(n_samples=500, random_seed=42 + i)
        filename = f'data/raw/p{i}_extrac.csv'
        df.to_csv(filename, index=False, sep=';')
        print(f"  ✓ {filename} ({len(df)} filas, {df.shape[1]} columnas)")
    
    print()
    print("=" * 60)
    print("✅ Generación completada")
    print("=" * 60)
    print()
    print("Siguientes pasos:")
    print("1. Instalar dependencias: pip install -r requirements.txt")
    print("2. Verificar instalación: python test_installation.py")
    print("3. Ejecutar pipeline: python main.py --mode training --input data/raw/")
    print()

if __name__ == '__main__':
    main()

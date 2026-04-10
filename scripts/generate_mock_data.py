import pandas as pd
import numpy as np
from pathlib import Path

def generate_mock_data(n_samples=1000):
    np.random.seed(42)
    
    # Sensor features matching your preprocess logic
    data = {
        'ip': np.random.uniform(0.1, 5.0, n_samples),
        'ep': np.random.uniform(10, 100, n_samples),
        'ph': np.random.uniform(6.5, 8.5, n_samples),
        'temp': np.random.uniform(36.0, 39.0, n_samples),
        'concentration': np.random.uniform(0.01, 1.0, n_samples),
        'cancer_risk_score': np.random.uniform(0, 100, n_samples),
        'months_to_onset': np.random.randint(1, 60, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Ensure raw directory exists
    output_path = Path("data/raw")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    csv_file = output_path / "sensor_data.csv"
    df.to_csv(csv_file, index=False)
    print(f"✅ Success: Mock data created with {n_samples} samples at {csv_file}")

if __name__ == "__main__":
    generate_mock_data()
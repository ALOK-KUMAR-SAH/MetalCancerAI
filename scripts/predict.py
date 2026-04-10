import torch
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from metalcancerai.modeling.pipeline import MetalCancerModel

def predict():
    # Paths setup
    repo_root = Path(__file__).resolve().parents[1]
    model_path = repo_root / "models" / "cancer_detector.pth"
    scaler_path = repo_root / "models" / "scaler.joblib"

    if not scaler_path.exists() or not model_path.exists():
        print("Error: Model artifacts (pth/joblib) missing! Please run training first.")
        return

    # 1. Load Preprocessor & Model
    print("🔄 Loading AI Model and Scaler...")
    preprocessor = joblib.load(scaler_path)
    
    # Determine input dim automatically from fitted preprocessor
    input_dim = preprocessor.transform(pd.DataFrame([ [0]*len(preprocessor.feature_names_in_) ], 
                                       columns=preprocessor.feature_names_in_)).shape[1]
    
    model = MetalCancerModel(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # 2. Sample Patient Data (Input for Prediction)
    # Aap yahan values change karke test kar sakte hain
    new_sample = pd.DataFrame({
        'ip': [0.65],           # Peak Current
        'ep': [55.2],           # Peak Potential
        'peak_width': [0.18], 
        'area': [1.42], 
        'baseline': [0.05],
        'ph': [7.35],           # Blood/Urine pH
        'temperature': [37.1],  # Body Temp
        'sample_type': ['urine'],
        'age': [58], 
        'sex': ['M'], 
        'diabetes': [1], 
        'ckd': [0], 
        'cvd': [1], 
        'cancer': [0], 
        'inflammation': [1]
    })

    # 3. Preprocessing
    X_scaled = preprocessor.transform(new_sample)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # 4. Inference
    with torch.no_grad():
        risk_raw, time_raw = model(X_tensor)

    # 5. Post-Processing (Normalizing high scores to realistic range)
    # Risk score ko 0-100% range mein map kar rahe hain
    final_risk = torch.sigmoid(risk_raw / 100).item() * 100 
    # Time ko positive months mein constrain kar rahe hain
    final_time = max(1, round(abs(time_raw.item() % 60))) 

    # 6. Final Report Display
    print("\n" + "="*40)
    print("      IIT PATNA - AI MEDICAL REPORT")
    print("="*40)
    print(f"  PATIENT AGE    : {new_sample['age'][0]}")
    print(f"  SAMPLE TYPE    : {new_sample['sample_type'][0].upper()}")
    print("-" * 40)
    print(f"  PREDICTED RISK : {final_risk:.2f}%")
    print(f"  EST. ONSET TIME: {final_time} Months")
    print("-" * 40)
    
    # Simple Logic-based Advice
    if final_risk > 70:
        print("  STATUS         : HIGH RISK - Urgent Clinical Review")
    elif final_risk > 30:
        print("  STATUS         : MODERATE - Regular Monitoring Needed")
    else:
        print("  STATUS         : LOW RISK - Routine Follow-up")
    print("="*40 + "\n")

if __name__ == "__main__":
    predict()
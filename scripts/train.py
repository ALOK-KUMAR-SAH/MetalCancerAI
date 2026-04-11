import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import joblib
from pathlib import Path
from metalcancerai.data.preprocess import preprocess_table
from metalcancerai.modeling.pipeline import MetalCancerModel

def train_model():
    # Load Data
    data_path = Path("data/raw/synthetic_medical_data.csv")
    df = pd.read_csv(data_path)
    
    # Preprocess
    X, y, preprocessor = preprocess_table(df)
    X_transformed = preprocessor.fit_transform(X)
    
    # Convert to Tensors
    X_tensor = torch.tensor(X_transformed, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    
    # Initialize Model
    model = MetalCancerModel(input_dim=X_tensor.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop (Brief)
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        risk_p, time_p = model(X_tensor)
        
        # Multi-task Loss
        loss_risk = criterion(risk_p, y_tensor[:, 0:1] / 100.0) # Scale target to 0-1
        loss_time = criterion(time_p, y_tensor[:, 1:2])
        
        total_loss = loss_risk + (0.1 * loss_time) # Weighting
        total_loss.backward()
        optimizer.step()
        
    # SAVE ARTIFACTS
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    # Fixed naming convention
    joblib.dump(preprocessor, model_dir / "preprocessor.joblib")
    torch.save(model.state_dict(), model_dir / "cancer_detector.pth")
    print("✅ Training complete. Artifacts saved as 'preprocessor.joblib' and 'cancer_detector.pth'")

if __name__ == "__main__":
    train_model()
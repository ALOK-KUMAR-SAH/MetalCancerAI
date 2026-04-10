import torch
import torch.nn as nn
import pandas as pd
import joblib
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from metalcancerai.data.preprocess import preprocess_table
from metalcancerai.modeling.pipeline import MetalCancerModel

def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / "raw" / "sensor_data.csv"
    model_dir = repo_root / "models"
    model_dir.mkdir(exist_ok=True)

    df = pd.read_csv(data_path)
    X_raw, y_df, preprocessor = preprocess_table(df)
    
    # Fit and Transform
    X_np = preprocessor.fit_transform(X_raw)
    
    # SAVE THE SCALER/PREPROCESSOR HERE
    joblib.dump(preprocessor, model_dir / "scaler.joblib")
    print(f"✅ Scaler saved at: {model_dir / 'scaler.joblib'}")

    # Convert to Tensors
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_df.to_numpy(), dtype=torch.float32)

    loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
    model = MetalCancerModel(input_dim=X.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print("Starting Training...")
    for epoch in range(1, 51):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            risk_p, time_p = model(xb)
            loss = criterion(risk_p.squeeze(), yb[:, 0]) + (0.1 * criterion(time_p.squeeze(), yb[:, 1]))
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), model_dir / "cancer_detector.pth")
    print("✅ Model weights saved!")

if __name__ == "__main__":
    main()
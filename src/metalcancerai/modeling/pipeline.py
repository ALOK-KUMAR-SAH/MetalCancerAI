import torch
import torch.nn as nn

class MetalCancerModel(nn.Module):
    def __init__(self, input_dim: int):
        """
        Multi-Task Learning Model for Metal-Induced Cancer Detection.
        
        Args:
            input_dim (int): Preprocessed features ki total count.
        """
        super(MetalCancerModel, self).__init__()
        
        # 1. Shared Backbone: Common features seekhne ke liye
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 2. Task-Specific Head A: Risk Score Prediction (Regression)
        self.risk_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # 3. Task-Specific Head B: Months to Onset Prediction (Regression)
        self.time_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor):
        # Shared representations extract karein
        shared_out = self.shared_layers(x)
        
        # Alag-alag outputs generate karein
        risk_score = self.risk_head(shared_out)
        months_to_onset = self.time_head(shared_out)
        
        return risk_score, months_to_onset
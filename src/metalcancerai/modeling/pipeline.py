import torch
import torch.nn as nn

class MetalCancerModel(nn.Module):
    def __init__(self, input_dim):
        super(MetalCancerModel, self).__init__()
        
        # Shared Layers (Backbone)
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Task 1: Risk Probability Head
        self.risk_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # Task 2: Time to Onset Head
        self.time_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        shared_out = self.shared_layers(x)
        
        # NEW: Constrain risk to 0-1 (will be multiplied by 100 in app)
        risk_score = torch.sigmoid(self.risk_head(shared_out))
        
        # NEW: Constrain time to be positive (ReLU ensures it's >= 0)
        months_to_onset = torch.relu(self.time_head(shared_out))
        
        return risk_score, months_to_onset
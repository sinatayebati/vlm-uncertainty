import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(64, 3)
        self.log_std_head = nn.Linear(64, 3)

    def forward(self, x):
        x = self.fc(x)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        std = torch.exp(log_std)
        return mean, std
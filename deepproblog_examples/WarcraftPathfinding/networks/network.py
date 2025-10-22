import torch
import torch.nn as nn

# Tiny CNN for 3x8x8 -> 5 classes
class TileCostCNN(nn.Module):
    def __init__(self, in_ch=3, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, num_classes),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        # DeepProbLog will pass [tile] as a list; be robust:
        if isinstance(x, (list, tuple)) and len(x)==1:
            x = x[0]
        # expect (B,C,8,8) or (C,8,8)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return self.net(x)  # (B,5)
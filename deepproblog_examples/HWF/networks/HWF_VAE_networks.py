import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim2, z_dim):
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(1, 6, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 45 -> 22
            nn.Conv2d(6, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 22 -> 11
        )
        self.mlp = nn.Sequential(
            nn.Linear(16 * 11 * 11, 128),
            nn.ReLU(),
            nn.Linear(128, 84),
            nn.ReLU(),
            nn.Linear(84, z_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        if isinstance(x, (list, tuple)) and len(x) == 1:
            x = x[0]
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        z = self.convolutions(x).view(x.size(0), -1)
        z = self.mlp(z)
        return z.view(1, -1)

class Reshape(nn.Module):
    def __init__(self, shape): super().__init__(); self.shape = shape
    def forward(self, x): return x.view(*self.shape)

class Decoder(nn.Module):
    def __init__(self, x_dim, h_dim2, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.Decoder = nn.Sequential(
            nn.Linear(z_dim, 16 * 11 * 11),
            Reshape((-1, 16, 11, 11)),
            nn.ReLU(),
            # 11 -> 22
            nn.ConvTranspose2d(
                16, 6, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            # 22 -> 45  (padding=0, output_padding=0)
            nn.ConvTranspose2d(
                6, 1, kernel_size=3, stride=2, padding=0, output_padding=0
            ),
            nn.Tanh(),
        )

    def forward(self, z):
        if isinstance(z, (list, tuple)) and len(z) == 1:
            z = z[0]
        z = z.view(-1, self.z_dim)
        z = self.Decoder(z)
        return z

def encoder(lat_dim=12):
    module = Encoder(x_dim=45*45, h_dim2=128, z_dim=lat_dim)
    opt = torch.optim.Adam(module.parameters(), lr=1e-4)
    return module, opt

def decoder(lat_dim=12):
    module = Decoder(x_dim=45*45, h_dim2=128, z_dim=lat_dim)
    opt = torch.optim.Adam(module.parameters(), lr=1e-4)
    return module, opt
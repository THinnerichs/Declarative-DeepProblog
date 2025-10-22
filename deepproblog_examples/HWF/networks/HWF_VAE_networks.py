import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Encoder ----------------
class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim2, z_dim):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 6, 3, stride=1, padding=1),  # 45x45 -> 45x45
            nn.ReLU(),
            nn.MaxPool2d(2),                          # 45 -> 22
            nn.Conv2d(6, 16, 3, stride=1, padding=1), # 22x22
            nn.ReLU(),
            nn.MaxPool2d(2),                          # 22 -> 11
        )
        self.mlp = nn.Sequential(
            nn.Linear(16 * 11 * 11, 128),
            nn.ReLU(),
            nn.Linear(128, z_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        # Accept list/tuple or tensor; ensure [B,1,45,45]
        if isinstance(x, (list, tuple)) and len(x) == 1:
            x = x[0]
        if x.dim() == 3:  # [1,45,45]
            x = x.unsqueeze(0)
        # x: [B,1,45,45]
        h = self.convs(x)                           # [B,16,11,11]
        h = h.view(h.size(0), -1)                   # [B, 16*11*11]
        z = self.mlp(h)                             # [B, z_dim]
        return z

# --------------- Small helper ---------------
class Reshape(nn.Module):
    def __init__(self, shape): super().__init__(); self.shape = shape
    def forward(self, x): return x.view(*self.shape)

# ---------------- Decoder (resize-conv, no deconv) ----------------
class Decoder(nn.Module):
    """
    11 -> (upsample x2) 22 -> conv -> 22
       -> (upsample x2) 44 -> conv(k=2,p=1) -> 45
    """
    def __init__(self, x_dim, h_dim2, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.fc = nn.Linear(z_dim, 16 * 11 * 11)
        self.dec = nn.Sequential(
            # start: [B,16,11,11]
            nn.Upsample(scale_factor=2, mode="nearest"),     # 11 -> 22
            nn.Conv2d(16, 8, kernel_size=3, padding=1), nn.ReLU(),  # 22 -> 22
            nn.Upsample(scale_factor=2, mode="nearest"),     # 22 -> 44
            # magic: kernel=2, padding=1 increases 44 -> 45 with stride=1
            nn.Conv2d(8, 1, kernel_size=2, padding=1),       # 44 -> 45
            nn.Tanh(),
        )

    def forward(self, z):
        if isinstance(z, (list, tuple)) and len(z) == 1:
            z = z[0]
        if z.dim() == 1:
            z = z.unsqueeze(0)
        z = z.view(-1, self.z_dim)               # [B, z_dim]
        h = self.fc(z).view(-1, 16, 11, 11)      # [B,16,11,11]
        x = self.dec(h)                           # [B,1,45,45]
        return x

# factory functions (unchanged signatures)
def encoder(lat_dim=12):
    module = Encoder(x_dim=45*45, h_dim2=128, z_dim=lat_dim)
    opt = torch.optim.Adam(module.parameters(), lr=1e-4)
    return module, opt

def decoder(lat_dim=12):
    module = Decoder(x_dim=45*45, h_dim2=128, z_dim=lat_dim)
    opt = torch.optim.Adam(module.parameters(), lr=1e-4)
    return module, opt

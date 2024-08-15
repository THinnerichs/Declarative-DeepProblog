import torch
from torch import nn


class SymbolEncoder(nn.Module):
    def __init__(self, embed_size):
        super(SymbolEncoder, self).__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(1, 6, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.4),
        )

        self.mlp = nn.Sequential(
            nn.Linear(16 * 11 * 11, embed_size),
            nn.Tanh()
            # nn.Dropout2d(0.8)
        )

    def forward(self, x):
        x = self.convolutions(x)
        x = torch.flatten(x, 1)
        x = self.mlp(x)
        return x


class SymbolDecoder(nn.Module):
    def __init__(self, embed_size):
        super(SymbolDecoder, self).__init__()
        # decoder part

        self.decoder = nn.Sequential(
            nn.Linear(embed_size, 128),
            nn.ReLU(),
            nn.Linear(128, 45 * 45),
            nn.Tanh()
        )

        # self.decoder = nn.Sequential(
        #     nn.Linear(12, 64 * 4 * 4),
        #     Reshape((-1, 64, 4, 4)),  # Reshape to (batch_size, channels, height, width)
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(4, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.Sigmoid()
        # )

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def forward(self, z):
        mu, log_var = torch.chunk(z, 2, dim=-1)
        z = self.sampling(mu, log_var)
        z =  self.decoder(z)
        z = z.view(-1, 1, 45, 45)
        return z

def encoder(lat_dim=12):
    module = SymbolEncoder(embed_size=lat_dim)
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-4)
    return module, optimizer


def decoder(lat_dim=12):
    module = SymbolDecoder(embed_size=lat_dim)
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-4)
    return module, optimizer
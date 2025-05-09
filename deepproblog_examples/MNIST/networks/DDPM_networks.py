import torch
import torch.nn as nn
import torch.nn.functional as F


# From https://github.com/lyeoni/pytorch-mnist-VAE/blob/master/pytorch-mnist-VAE.ipynb

class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim2, z_dim):
        super(Encoder, self).__init__()

        # Convolutional network
        self.convolutions = nn.Sequential(
            nn.Conv2d(1, 6, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Fully connected network
        self.mlp = nn.Sequential(
            nn.Linear(16 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 84),
            nn.ReLU(),
            nn.Linear(84, z_dim),
            nn.Tanh()
            # nn.Dropout2d(0.8)
        )

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = self.fc2(h)
        return h

    def forward(self, x):
        x = x[0]
        # z = self.encoder(x.view(-1, 784))
        x = x.view(-1,1,28,28)
        z = self.convolutions(x).view(-1, 16*7*7)
        z = self.mlp(z).view(1, -1)
        return z


class Reshape(nn.Module):
    """
    A module that reshapes its input tensor to a specified shape.
    """
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class Decoder(nn.Module):
    def __init__(self, x_dim, h_dim2, z_dim):
        super(Decoder, self).__init__()
        self.z_dim = z_dim

        # decoder part
        self.latent_to_feature = nn.Linear(z_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        z = z[0].view(-1, self.z_dim)

        # Generate noise input (like DDPM's x_t)
        # x_t = torch.randn(z.size(0), 1, 28, 28, device=z.device)

        z_feat = self.latent_to_feature(z).view(-1, 64, 7, 7)
        # z_upsampled = F.interpolate(z_feat, size=(28, 28), mode='bilinear')
        x = z_feat

        # print("z_upsampled.shape:", z_upsampled.shape)
        # x = torch.cat([x_t, z_upsampled], dim=1)

        h = self.decoder(x)
        h = h.view(-1, 1, 28, 28)
        return h


def encoder(lat_dim=12):
    module = Encoder(x_dim=784, h_dim2=128, z_dim=lat_dim)
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-4)
    return module, optimizer


def decoder(lat_dim=12):
    module = Decoder(x_dim=784, h_dim2=128, z_dim=lat_dim)
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-4)
    return module, optimizer

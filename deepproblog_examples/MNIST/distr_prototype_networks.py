import torch
import torch.nn as nn
import torch.nn.functional as F


# From https://github.com/lyeoni/pytorch-mnist-VAE/blob/master/pytorch-mnist-VAE.ipynb


class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim2, z_dim):
        super(Encoder, self).__init__()

        # encoder part
        # self.fc1 = nn.Linear(x_dim, h_dim2)
        # self.fc2 = nn.Linear(h_dim2, z_dim)

        self.convolutions = nn.Sequential(
            nn.Conv2d(1, 6, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.mlp = nn.Sequential(
            nn.Linear(16 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 12),
            # nn.ReLU(),
            # nn.Linear(84, 12),
            nn.Tanh()
            # nn.Dropout2d(0.8)
        )

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h)

    def forward(self, x):
        print(f"x.shape: {x.shape}")
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
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc6 = nn.Linear(h_dim2, x_dim)
        self.z_dim = z_dim

        # self.Decoder = nn.Sequential(
        #     nn.Linear(z_dim, 32 * 7 * 7),
        #     Reshape((-1, 32, 7, 7)),  # Reshape to (batch_size, channels, height, width)
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=1),
        #     nn.Tanh()
        # )

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample


    def decoder(self, z):
        if z.size(1) == self.z_dim * 2:
            mu, log_var = torch.chunk(z, 2, dim=-1)
            z = self.sampling(mu, log_var)

        # h = self.Decoder(z)

        # # z = torch.stack(z)
        h = F.relu(self.fc4(z))
        h = self.fc6(h)
        h = torch.tanh(h)
        # # h = h.view(-1, 28, 28)
        h = h.view(-1, 1, 28, 28)
        return h

    def forward(self, z):
        return self.decoder(z)


def encoder(lat_dim=12):
    module = Encoder(x_dim=784, h_dim2=128, z_dim=lat_dim)
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-4)
    return module, optimizer


def decoder(lat_dim=12):
    module = Decoder(x_dim=784, h_dim2=128, z_dim=lat_dim)
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-4)
    return module, optimizer

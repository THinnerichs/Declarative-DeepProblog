import torch
import torch.nn as nn
import torch.nn.functional as F


# From https://github.com/lyeoni/pytorch-mnist-VAE/blob/master/pytorch-mnist-VAE.ipynb

class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(Encoder, self).__init__()

        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim2)
        # self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        # self.fc32 = nn.Linear(h_dim2, z_dim)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        # h = F.relu(self.fc2(h))
        #  return self.fc31(h), self.fc32(h)  # mu, log_var # VAE
        return self.fc31(h) # AE


    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def forward(self, x):
        x = torch.stack(x)
        # mu, log_var = self.encoder(x.view(-1, 784))
        z = self.encoder(x.view(-1, 784)) # AE
        # z = self.sampling(mu, log_var) 
        return z


class Decoder(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(Decoder, self).__init__()
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        # self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim2, x_dim)

    def decoder(self, z):
        z = torch.stack(z)
        h = F.relu(self.fc4(z))
        # h = F.relu(self.fc5(h))
        h = torch.tanh(self.fc6(h))
        h = h.view(-1, 1, 28, 28)
        return h

    def forward(self, z):
        return self.decoder(z)


def encoder():
    module = Encoder(x_dim=784, h_dim1=256, h_dim2=128, z_dim=12)
    optimizer = torch.optim.Adam(module.parameters())
    return module, optimizer


def decoder():
    module = Decoder(x_dim=784, h_dim1=256, h_dim2=128, z_dim=12)
    optimizer = torch.optim.Adam(module.parameters())
    return module, optimizer


class LossFunc(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
    
    def forward(self, x, mu, log_var):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD

def loss_func():
    module = LossFunc()
    optimizer = torch.optim.Adam(module.parameters())
    return module, optimizer


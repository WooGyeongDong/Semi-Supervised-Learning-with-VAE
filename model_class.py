import torch
import torch.nn as nn
import torch.nn.functional as F

# def reparameterization(mu, logvar):
#     std = torch.exp(logvar/2)
#     eps = torch.ones_like(std, dtype=torch.float32)
#     return mu + eps * std

#%%
class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__()

        # 1st hidden layer
        self.fc1 = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.Softplus()
        )

        # output layer
        self.mu = nn.Linear(h_dim, z_dim)
        self.logvar = nn.Linear(h_dim, z_dim)

    def forward(self, x):
        x = self.fc1(x)

        mu = self.mu(x)
        logvar = self.logvar(x)

        z = reparameterization(mu, logvar)
        return z, mu, logvar
    
#%%
class Decoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__()

        # 1st hidden layer
        self.fc1 = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.Softplus()
        )

        # output layer
        self.fc2 = nn.Linear(h_dim, x_dim)

    def forward(self, z):
        z = self.fc1(z)
        x_reconst = torch.sigmoid(self.fc2(z))
        return x_reconst

#%%
class VAE(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__()
        self.encoder = Encoder(x_dim, h_dim, z_dim)
        self.decoder = Decoder(x_dim, h_dim, z_dim)
        
    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        x_reconst = self.decoder(z)
        return x_reconst, mu, logvar 
    
def loss_func(x, x_reconst, mu, logvar):
    kl_div = 0.5 * torch.sum(mu**2 + logvar.exp() - logvar - 1)
    reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
    loss = kl_div + reconst_loss
    
    return loss

'''
M2
'''

class Encoder2(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__()

        # 1st hidden layer
        self.fc1 = nn.Sequential(
            nn.Linear(x_dim+10, h_dim),
            nn.Softplus()
        )

        # output layer
        self.mu = nn.Linear(h_dim, z_dim)
        self.logvar = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, z_dim)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, 10)
        )

    def forward(self, x, label):
        pi = self.fc2(x)
        y = torch.softmax(pi, dim = 1)

        xy = torch.cat([x, label], dim = -1)
        xy = self.fc1(xy)
        mu = self.mu(xy)
        logvar = self.logvar(x)

        z = reparameterization(mu, logvar)
        return z, mu, logvar, y
    
#%%
class Decoder2(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__()

        # 1st hidden layer
        self.fc1 = nn.Sequential(
            nn.Linear(z_dim + 10, h_dim),
            nn.Softplus()
        )

        # output layer
        self.fc2 = nn.Linear(h_dim, x_dim)

    def forward(self, z, label):
        z = torch.cat([z, label], dim = -1)
        z = self.fc1(z)
        x_reconst = torch.sigmoid(self.fc2(z))
        return x_reconst

#%%
class VAE2(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__()
        self.encoder = Encoder2(x_dim, h_dim, z_dim)
        self.decoder = Decoder2(x_dim, h_dim, z_dim)
        
    def forward(self, x, label):
        z, mu, logvar, y = self.encoder(x, label)
        x_reconst = self.decoder(z, label)
        return x_reconst, mu, logvar, y

def log_standard_categorical(p):
    prior = F.softmax(torch.ones_like(p), dim=1)
    prior.requires_grad = False

    cross_entropy = -torch.sum(p * torch.log(prior + 1e-8), dim=1)

    return cross_entropy    

def loss_func2(x, x_reconst, mu, logvar, target, N, y = None):
    if y == None : no_label = True 
    else: no_label = False
    kl_div = 0.5 * torch.sum(mu**2 + logvar.exp() - logvar - 1)
    reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
    prior = log_standard_categorical(y)
    classification_loss = F.cross_entropy(y, target, reduction='sum')*0.1*N
    L = kl_div + reconst_loss + prior
    if not no_label: 
        return L + classification_loss
    H = torch.sum(-y*y.log())
    U = L + H
    return U + classification_loss


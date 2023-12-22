import torch
import torch.nn as nn
import torch.nn.functional as F

def reparameterization(mu, logvar):
    std = torch.exp(logvar/2)
    eps = torch.randn_like(std, dtype=torch.float32)
    return mu + eps * std
#%%
####   M1

class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__()

        # 1st hidden layer
        self.fc1 = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.Tanh(),
            # nn.Linear(h_dim, h_dim),
            # nn.Tanh() 
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
    
class Decoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__()

        # 1st hidden layer
        self.fc1 = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus()
        )

        # output layer
        self.fc2 = nn.Linear(h_dim, x_dim)

    def forward(self, z):
        z = self.fc1(z)
        x_reconst = torch.sigmoid(self.fc2(z))
        return x_reconst

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

################################################################################

###   M2

#%%
class Classifier(nn.Module):
    def __init__(self, x_dim, h_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, 10)
        )

    def forward(self, x):
        pi = self.fc(x)
        y = torch.softmax(pi, dim = 1)
        return y

#%%
class Encoder2(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__()

        # 1st hidden layer
        self.fc1 = nn.Sequential(
            nn.Linear(x_dim+10, h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus()
        )

        # output layer
        self.mu = nn.Linear(h_dim, z_dim)
        self.logvar = nn.Linear(h_dim, z_dim)

    def forward(self, x):
        x = self.fc1(x)
        mu = self.mu(x)
        logvar = F.softplus(self.logvar(x))

        z = reparameterization(mu, logvar)
        return z, mu, logvar
    
#%%
class Decoder2(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__()

        # 1st hidden layer
        self.fc1 = nn.Sequential(
            nn.Linear(z_dim + 10, h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus()
        )

        # output layer
        self.fc2 = nn.Linear(h_dim, x_dim)

    def forward(self, z):
        z = self.fc1(z)
        x_reconst = torch.sigmoid(self.fc2(z))
        return x_reconst

#%%
class VAE2(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__()
        self.encoder = Encoder2(x_dim, h_dim, z_dim)
        self.decoder = Decoder2(x_dim, h_dim, z_dim)
        self.classifier = Classifier(x_dim, h_dim)
        self.apply(self.init_weights)
        
    def forward(self, x, label):
        z, mu, logvar = self.encoder(torch.cat([x, label], dim = -1))
        x_reconst = self.decoder(torch.cat([z, label], dim = -1))
        return x_reconst, mu, logvar
    
    def classify(self, x):
        y = self.classifier(x)
        return y
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean = 0.0, std = 0.001)
            nn.init.constant_(module.bias, 0)


#################################################################

    
class VAE12(nn.Module):
    def __init__(self, z1_dim, h_dim, z2_dim, x_dim):
        super().__init__()
        self.encoder = Encoder2(z1_dim, h_dim, z2_dim)
        self.decoder = Decoder2(x_dim, h_dim, z2_dim)
        self.classifier = Classifier(z1_dim, h_dim)
        self.apply(self.init_weights)
        
    def forward(self, x, label):
        z, mu, logvar = self.encoder(torch.cat([x, label], dim = -1))
        x_reconst = self.decoder(torch.cat([z, label], dim = -1))
        return x_reconst, mu, logvar
    
    def classify(self, x):
        y = self.classifier(x)
        return y
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean = 0.0, std = 0.001)
            nn.init.constant_(module.bias, 0)
    
class Decoder_norm2(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__()

        # 1st hidden layer
        self.fc1 = nn.Sequential(
            nn.Linear(z_dim+10, h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus()
        )

        # output layer
        self.mu = nn.Linear(h_dim, x_dim)
        self.logvar = nn.Linear(h_dim, x_dim)

    def forward(self, z):
        z = self.fc1(z)
        
        mu_de = torch.sigmoid(self.mu(z))
        logvar_de = self.logvar(z)
        
        x_reconst = reparameterization(mu_de, logvar_de)
        return x_reconst, mu_de, logvar_de

class VAE123(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__()
        self.encoder = Encoder2(x_dim, h_dim, z_dim)
        self.decoder = Decoder_norm2(x_dim, h_dim, z_dim)
        self.classifier = Classifier(x_dim, h_dim)
        
    def forward(self, x, label):
        z, mu, logvar = self.encoder(torch.cat([x, label], dim = -1))
        x_reconst, mu_de, logvar_de = self.decoder(torch.cat([z, label], dim = -1))
        return x_reconst, mu_de, logvar_de, mu, logvar
    
    def classify(self, x):
        y = self.classifier(x)
        return y
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean = 0.0, std = 0.001)
            nn.init.constant_(module.bias, 0)


class Decoder_norm(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__()

        # 1st hidden layer
        self.fc1 = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.Softplus(),
            # nn.Linear(h_dim, h_dim),
            # nn.Softplus()
        )

        # output layer
        self.mu = nn.Linear(h_dim, x_dim)
        self.logvar = nn.Linear(h_dim, x_dim)

    def forward(self, z):
        z = self.fc1(z)
        
        mu_de = torch.sigmoid(self.mu(z))
        logvar_de = self.logvar(z)
        
        x_reconst = reparameterization(mu_de, logvar_de)
        return x_reconst, mu_de, logvar_de
    
class VAE_norm(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__()
        self.encoder = Encoder(x_dim, h_dim, z_dim)
        self.decoder = Decoder_norm(x_dim, h_dim, z_dim)
        
    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        x_reconst, mu_de, logvar_de = self.decoder(z)
        return x_reconst, mu_de, logvar_de, mu, logvar
    
def loss_norm(x, mu_de, logvar_de , mu, logvar):
    kl_div = 0.5 * torch.sum(mu**2 + logvar.exp() - logvar - 1)
    reconst_loss = 0.5*torch.sum(((x-mu_de)**2)/torch.exp(logvar_de)+logvar_de)
    loss = kl_div + reconst_loss
    
    return loss
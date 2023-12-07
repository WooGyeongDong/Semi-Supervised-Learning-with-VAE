import torch
import torch.nn as nn
import torch.nn.functional as F

# def reparameterization(mu, logvar):
#     std = torch.exp(logvar/2)
#     eps = torch.ones_like(std, dtype=torch.float32)
#     return mu + eps * std
def reparameterization(mu, logvar):
    std = torch.exp(logvar/2)
    eps = torch.randn(mu.size())
    return mu.addcmul(std, eps)
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

class Classifier2(nn.Module):
    def __init__(self, x_dim, h_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
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
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
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
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
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
        self.classifier = Classifier2(x_dim, h_dim)
        
    def forward(self, x, label):
        z, mu, logvar = self.encoder(torch.cat([x, label], dim = -1))
        x_reconst = self.decoder(torch.cat([z, label], dim = -1))
        return x_reconst, mu, logvar
    
    def classify(self, x):
        y = self.classifier(x)
        return y

#################################################################


class Classifier(nn.Module):
    def __init__(self, dims):
        """
        Single hidden layer classifier
        with softmax output.
        """
        super(Classifier, self).__init__()
        [x_dim, h_dim, y_dim] = dims
        self.dense = nn.Linear(x_dim, h_dim)
        self.logits = nn.Linear(h_dim, y_dim)

    def forward(self, x):
        x = F.relu(self.dense(x))
        x = F.softmax(self.logits(x), dim=-1)
        return x
    
     
class GaussianSample(nn.Module):
    """
    Layer that represents a sample from a
    Gaussian distribution.
    """
    def __init__(self, in_features, out_features):
        super(GaussianSample, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu = nn.Linear(in_features, out_features)
        self.log_var = nn.Linear(in_features, out_features)

    def reparametrize(self, mu, log_var):
        epsilon = torch.randn(mu.size())

        if mu.is_cuda:
            epsilon = epsilon.cuda()

        # log_std = 0.5 * log_var
        # std = exp(log_std)
        std = log_var.mul(0.5).exp_()

        # z = std * epsilon + mu
        z = mu.addcmul(std, epsilon)

        return z
    
    def forward(self, x):
        mu = self.mu(x)
        log_var = F.softplus(self.log_var(x))

        # return self.reparametrize(mu, log_var), mu, log_var
        return reparameterization(mu, log_var), mu, log_var
    
#%%

    
class Encoder(nn.Module):
    def __init__(self, dims, sample_layer=GaussianSample):
 
        super(Encoder, self).__init__()

        [x_dim, h_dim, z_dim] = dims
        neurons = [x_dim, *h_dim]
        linear_layers = [nn.Linear(neurons[i-1], neurons[i]) for i in range(1, len(neurons))]

        self.hidden = nn.ModuleList(linear_layers)
        self.sample = sample_layer(h_dim[-1], z_dim)

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.sample(x)


class Decoder(nn.Module):
    def __init__(self, dims):

        super(Decoder, self).__init__()

        [z_dim, h_dim, x_dim] = dims

        neurons = [z_dim, *h_dim]
        linear_layers = [nn.Linear(neurons[i-1], neurons[i]) for i in range(1, len(neurons))]
        self.hidden = nn.ModuleList(linear_layers)

        self.reconstruction = nn.Linear(h_dim[-1], x_dim)

        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.output_activation(self.reconstruction(x))


class VariationalAutoencoder(nn.Module):
    def __init__(self, dims):

        super(VariationalAutoencoder, self).__init__()

        [x_dim, z_dim, h_dim] = dims
        self.z_dim = z_dim

        self.encoder = Encoder([x_dim, h_dim, z_dim])
        self.decoder = Decoder([z_dim, list(reversed(h_dim)), x_dim])
        self.kl_divergence = 0

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x, y=None):

        z, z_mu, z_log_var = self.encoder(x)

        # self.kl_divergence = kld(z_mu, z_log_var)

        x_mu = self.decoder(z)

        return x_mu


class DeepGenerativeModel(VariationalAutoencoder):
    def __init__(self, dims):

        [x_dim, self.y_dim, z_dim, h_dim] = dims
        super(DeepGenerativeModel, self).__init__([x_dim, z_dim, h_dim])

        self.encoder = Encoder([x_dim + self.y_dim, h_dim, z_dim])
        self.decoder = Decoder([z_dim + self.y_dim, list(reversed(h_dim)), x_dim])
        self.classifier = Classifier([x_dim, h_dim[0], self.y_dim])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y):
        # Add label and data and generate latent variable
        z, z_mu, z_log_var = self.encoder(torch.cat([x, y], dim=-1))

        # self.kl_divergence = self.kld(z_mu, z_log_var)

        # Reconstruct data point from latent data and label
        x_mu = self.decoder(torch.cat([z, y], dim=1))

        return x_mu, z_mu, z_log_var

    def classify(self, x):
        logits = self.classifier(x)
        return logits


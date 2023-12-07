#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from itertools import cycle
import numpy as np
import math
import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
from data import load_MNIST
import random
#%%
def enumerate_discrete(x, y_dim):

    def batch(batch_size, label):
        labels = (torch.ones(batch_size, 1) * label).type(torch.LongTensor)
        y = torch.zeros((batch_size, y_dim))
        y.scatter_(1, labels, 1)
        return y.type(torch.LongTensor)

    batch_size = x.size(0)
    generated = torch.cat([batch(batch_size, i) for i in range(y_dim)])

    if x.is_cuda:
        generated = generated.cuda()

    return Variable(generated.float())

def log_standard_categorical(p):

    # Uniform prior over y
    prior = F.softmax(torch.ones_like(p), dim=1)
    prior.requires_grad = False

    cross_entropy = -torch.sum(p * torch.log(prior + 1e-8), dim=1)

    return cross_entropy

def kld(mu, log_var):
    kl = 0.5 * (mu**2 + log_var.exp() - log_var - 1)

    return torch.sum(kl, dim=-1)


#%%
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
    
def reparameterization(mu, logvar):
    std = torch.exp(logvar/2)
    eps = torch.randn(mu.size())
    return mu.addcmul(std, eps)
     
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
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x, y=None):

        z, z_mu, z_log_var = self.encoder(x)

        self.kl_divergence = kld(z_mu, z_log_var)

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
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y):
        # Add label and data and generate latent variable
        z, z_mu, z_log_var = self.encoder(torch.cat([x, y], dim=-1))

        # self.kl_divergence = self._kld(z, (z_mu, z_log_var))

        # Reconstruct data point from latent data and label
        x_mu = self.decoder(torch.cat([z, y], dim=1))

        return x_mu, z_mu, z_log_var

    def classify(self, x):
        logits = self.classifier(x)
        return logits

#%%
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
    
#%%

torch.manual_seed(1337)
np.random.seed(1337)

def binary_cross_entropy(r, x):
    "Drop in replacement until PyTorch adds `reduce` keyword."
    return -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)

n_labels = 10

train_data, test_data = load_MNIST()

indices = list(range(len(train_data)))
random.shuffle(indices)
valid_indices = indices[:10000]
labelled_indices = indices[10000:10000+3000]
unlabelled_indices = indices[10000+3000:]

labelled_batch_size = int(3000*100/50000)

labelled = DataLoader(train_data, batch_size=labelled_batch_size, pin_memory=True,
                                           sampler=SubsetRandomSampler(labelled_indices))
unlabelled = DataLoader(train_data, batch_size=100-labelled_batch_size, pin_memory=True,
                                            sampler=SubsetRandomSampler(unlabelled_indices))
validation = DataLoader(train_data, batch_size=100, pin_memory=True,
                                            sampler=SubsetRandomSampler(valid_indices))

alpha = 0.1 * 3000
#%%

# Kingma 2014, M2 model. Reported: 88%, achieved: ??%
# from models import DeepGenerativeModel
model = DeepGenerativeModel([784, n_labels, 2, [600, 600]])
model = VAE2(784, 500, 2)

likelihood=binary_cross_entropy
# elbo = SVI(model, likelihood=binary_cross_entropy)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))

epochs = 100
best = 0.0

def one_hot(digit):
    vector = torch.zeros(10)
    vector[digit] = 1
    return vector
#%%
for epoch in tqdm.tqdm(range(epochs)):
    model.train()
    for (x, y), (u, _) in zip(labelled, unlabelled):
        # Wrap in variables
        # x, y, u = Variable(x), Variable(y), Variable(u)
        y = torch.stack([one_hot(i) for i in y])
        # forward
        x = x.view(-1, 784)
        u = u.view(-1, 784)
        # L = -elbo(x, y)
        # U = -elbo(u)
        
        reconstruction, mu, log_var = model(x, y)
        # p(x|y,z)
        likeli = -likelihood(reconstruction, x)
        # p(y)
        prior = -log_standard_categorical(y)
        # Equivalent to -L(x, y)
        elbo = likeli + prior + kld(mu, log_var)
        elbo = -torch.mean(elbo)

        ys = enumerate_discrete(u, 10)
        ru = u.repeat(10, 1)
        logits = model.classify(u)
        reconstruction, mu, log_var = model(ru, ys)
        # p(x|y,z)
        likeli = -likelihood(reconstruction, ru)
        
        # p(y)
        prior = -log_standard_categorical(ys)
        # Equivalent to -L(x, y)
        L = likeli + prior + kld(mu, log_var)

        L = L.view_as(logits.t()).t()
        logits.shape
        L.shape
        # Calculate entropy H(q(y|x)) and sum over all labels
        H = -torch.sum(torch.mul(logits, torch.log(logits + 1e-8)), dim=-1)
        L = torch.sum(torch.mul(logits, L), dim=-1)

        # Equivalent to -U(x)
        U = L + H
        U = -torch.mean(U)

        # Add auxiliary classification loss q(y|x)
        logit = model.classify(x)
        classication_loss = torch.sum(y * torch.log(logit + 1e-8), dim=1).mean()

        J_alpha = elbo - alpha * classication_loss + U

        J_alpha.backward()
        optimizer.step()
        optimizer.zero_grad()
    # if epoch % 10 == 0:
        print(J_alpha)
           
# %%
import torchvision.utils
import matplotlib.pyplot as plt


def generate_grid(dim, grid_size, grid_range):
    """
    dim: 차원 수
    grid_size: 그리드 크기
    grid_range: 그리드의 범위, (시작, 끝)
    """
    grid = []
    for i in range(dim):
        axis_values = np.linspace(grid_range[0], grid_range[1], grid_size)
        grid.append(axis_values)

    grid_points = np.meshgrid(*grid)
    grid_points = np.column_stack([point.ravel() for point in grid_points])

    return grid_points

grid = generate_grid(2, 10, (-5,5))

latent_image = [model.decoder(torch.cat([torch.FloatTensor(i), 
                              torch.FloatTensor([0,0,0,0,1,0,0,0,0,0])])).reshape(-1,28,28) 
                              for i in grid]
latent_grid_img = torchvision.utils.make_grid(latent_image, nrow=10)
plt.imshow(latent_grid_img.permute(1,2,0))
plt.show()
# %%

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import tqdm
import importlib
from data import load_semi_MNIST
import model_class as mod
importlib.reload(mod)

#%%

torch.manual_seed(1337)
np.random.seed(1337)
labelled, unlabelled, validation = load_semi_MNIST(batch_size=100, labelled_size=3000)

#%%


def log_prior(p):
    prior = F.softmax(torch.ones_like(p), dim=-1)
    prior.requires_grad = False
    cross_entropy = -torch.sum(p * torch.log(prior), dim = -1)

    return cross_entropy

def kld(mu, logvar):
    kl = 0.5 * (mu**2 + logvar.exp() - logvar - 1)

    return torch.sum(kl, dim=-1)

def onehot(digit):
    vector = torch.zeros(10)
    vector[digit] = 1
    return vector
    

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

n_labels = 10

alpha = 0.1 * 3000
#%%

# Kingma 2014, M2 model. Reported: 88%, achieved: ??%
# from models import DeepGenerativeModel
# model = mod.DeepGenerativeModel([784, n_labels, 2, [600, 600]])
model = mod.VAE2(784, 500, 2)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))
for param in model.parameters():
    torch.nn.init.normal_(param, 0, 0.001)
epochs = 2

def loss_function(x, x_reconst, mu, logvar, label):
    kl_div = kld(mu, logvar)
    reconst_loss = torch.sum(F.binary_cross_entropy(x_reconst, x, reduction='none'), dim = -1)
    prior = log_prior(label)
    L = kl_div + reconst_loss + prior
    
    return L

#%%
for epoch in tqdm.tqdm(range(epochs)):
    model.train()
    for (x, target), (u, _) in zip(labelled, unlabelled):

        label = torch.stack([onehot(i) for i in target])
        x = x.view(-1, 784)
        u = u.view(-1, 784)
        
        x_reconst, mu, log_var = model(x, label)
      
        elbo = torch.mean(loss_function(x, x_reconst, mu, log_var, label))

        logits = model.classify(u)
        extend_y = torch.cat([torch.nn.functional.one_hot(torch.zeros(len(u)).long() + i, num_classes=10) for i in range(10)], dim=0).float()
        extend_u = u.repeat(10, 1)
        
        reconstruction, mu, log_var = model(extend_u, extend_y)
        
        L = loss_function(extend_u, reconstruction, mu, log_var, extend_y)
        L = L.view_as(logits.t()).t()

        # Calculate entropy H(q(y|x)) and sum over all label
        L = torch.sum(torch.mul(logits, L), dim=-1)
        H = torch.sum(torch.mul(logits, torch.log(logits + 1e-8)), dim=-1)

        # Equivalent to -U(x)
        # U = L + H
        # U = torch.mean(U)

        # # Add auxiliary classification loss q(y|x)
        # logit = model.classify(x)
        # classication_loss = -torch.sum(label * torch.log(logit + 1e-8), dim=1).mean()
        J = elbo + torch.mean(L + H)

        #Classification loss
        prob = model.classify(x)
        # classification_loss = F.cross_entropy(prob, label, reduction='mean')*0.1*config['labelled_size']
        classification_loss = -torch.sum(label * torch.log(prob + 1e-8), dim=1).mean()*300

        J_alpha = J + classification_loss

        # J_alpha = elbo + alpha * classication_loss + U
        optimizer.zero_grad()
        J_alpha.backward()
        optimizer.step()
        
    # if epoch % 10 == 0:
        print(J_alpha)
           


#%%
grid = generate_grid(2, 10, (-5,5))
latent_image = [model.decoder(torch.cat([torch.FloatTensor(i), 
                              torch.FloatTensor(onehot(4))])).reshape(-1,28,28) 
                              for i in grid]
latent_grid_img = torchvision.utils.make_grid(latent_image, nrow=10)
plt.imshow(latent_grid_img.permute(1,2,0))
plt.show()
# %%

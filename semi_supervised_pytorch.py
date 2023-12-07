#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import tqdm
import importlib
from data import load_semi_MNIST
import model_class as mod
importlib.reload(mod)
#%%

labelled, unlabelled, validation = load_semi_MNIST(100, 3000)

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

def kld(mu, logvar):
    kl = 0.5 * (mu**2 + logvar.exp() - logvar - 1)

    return torch.sum(kl, dim=-1)
    
#%%

torch.manual_seed(1337)
np.random.seed(1337)

def binary_cross_entropy(r, x):
    "Drop in replacement until PyTorch adds `reduce` keyword."
    return -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)

n_labels = 10

alpha = 0.1 * 3000
#%%

# Kingma 2014, M2 model. Reported: 88%, achieved: ??%
# from models import DeepGenerativeModel
# model = mod.DeepGenerativeModel([784, n_labels, 2, [600, 600]])
model = mod.VAE2(784, 500, 2)

likelihood=binary_cross_entropy
# elbo = SVI(model, likelihood=binary_cross_entropy)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))

epochs = 2
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
        elbo = likeli + prior - kld(mu, log_var)
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
        L = likeli + prior - kld(mu, log_var)

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

#%%

latent_image = [model.decoder(torch.cat([torch.FloatTensor(i), 
                              torch.FloatTensor(one_hot(7))])).reshape(-1,28,28) 
                              for i in grid]
latent_grid_img = torchvision.utils.make_grid(latent_image, nrow=10)
plt.imshow(latent_grid_img.permute(1,2,0))
plt.show()
# %%

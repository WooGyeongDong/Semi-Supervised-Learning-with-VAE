#%%
import torch
import torch.optim 
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import copy
import wandb
import importlib
from data import load_CIFAR10
import model_class as mod
importlib.reload(mod)
#%%
config = {'input_dim' : 3*32*32,
          'hidden_dim' : 1024,
          'latent_dim' : 500,
          'batch_size' : 500,
          'labelled_size' : 3000,
          'epochs' : 200,
          'lr' : 0.0003,
          'best_loss' : 10**9,
          'patience_limit' : 3}

# set seed
torch.manual_seed(23)

#%%
wb_log = True
if wb_log: wandb.init(project="cifar10", config=config)
is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')
print('Current cuda device is', device)
#%%
labelled, unlabelled, test_loader = load_CIFAR10(config['batch_size'], 
                                                config['labelled_size'], 
                                                seed_value=23)

#%%
def kld(mu, logvar):
    kl = 0.5 * (mu**2 + logvar.exp() - logvar - 1)

    return torch.sum(kl, dim=-1)

def log_prior(p):
    prior = F.softmax(torch.ones_like(p), dim=-1)
    prior.requires_grad = False
    cross_entropy = -torch.sum(p * torch.log(prior), dim = -1)

    return cross_entropy    


def elbo(x, mu_de, logvar_de , mu, logvar, label):
    kl_div = kld(mu, logvar)
    reconst_loss = 0.5*torch.sum(((x-mu_de)**2)/torch.exp(logvar_de)+logvar_de, dim = -1)
    prior = log_prior(label)
    L = kl_div + reconst_loss + prior
    
    return L

def onehot(digit):
    vector = torch.zeros(10)
    vector[digit] = 1
    return vector

def loss_function(x, label, u, model):
    # labelled data loss
    x_reconst, mu_de, logvar_de, mu, logvar = model(x, label)
    L = torch.mean(elbo(x, mu_de, logvar_de , mu, logvar, label))
    
    # unlabelled data loss
    u_prob = model.classify(u)
    temp_label = torch.cat([F.one_hot(torch.zeros(len(u)).long() + i, num_classes=10) for i in range(10)], dim=0).float().to(device)
    extend_u = u.repeat(10, 1)

    u_reconst, u_mu_de, u_logvar_de, u_mu, u_logvar = model(extend_u, temp_label)

    u_elbo = elbo(extend_u, u_mu_de, u_logvar_de, u_mu, u_logvar, temp_label)
    u_elbo = u_elbo.view_as(u_prob.t()).t()
    
    U = torch.sum(torch.mul(u_prob, u_elbo), dim = -1)
    H = torch.sum(torch.mul(u_prob, torch.log(u_prob + 1e-8)), dim = -1)
    
    J = L + torch.mean(U + H)

    #Classification loss
    prob = model.classify(x)
    classification_loss = -torch.sum(label * torch.log(prob + 1e-8), dim=1).mean()*0.1*config['labelled_size']

    loss = J + classification_loss
    return loss

#%%

model = mod.VAE123(x_dim=config['input_dim'], h_dim = config['hidden_dim'], z_dim = config['latent_dim']).to(device)
# optimizer = torch.optim.RMSprop(model.parameters(), lr = config['lr'], momentum=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))


#%%
img_size = config['input_dim']  
best_loss = config['best_loss']
patience_limit = config['patience_limit']
patience_check = 0 # 현재 몇 epoch 연속으로 loss 개선이 안되는지를 기록
val = []
for epoch in tqdm(range(config['epochs'])):
    model.train()
    train_loss = 0
    for (x, target), (u, _) in zip(labelled, unlabelled):
        # data processing
        label = torch.stack([onehot(i) for i in target]).to(device)
        x = x.view(-1, img_size).to(device)
        u = u.view(-1, img_size).to(device)

        loss = loss_function(x, label, u, model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print('Epoch: {} Train_Loss: {} :'.format(epoch, train_loss/len(labelled)))    

    with torch.no_grad(): 
        accuracy = 0
        for x, label in test_loader:
            x = x.view(-1, img_size).to(device)
            pred_idx = torch.argmax(model.classify(x), dim=-1)
            accuracy += torch.mean((pred_idx.data.to(device) == label.to(device)).float())
        print(f'{accuracy.item()/len(test_loader)*100:.2f}%') 
        if wb_log: wandb.log({'train_loss':train_loss/len(labelled), "Accuracy": accuracy.item()/len(test_loader)*100})
# %%
        
model_scripted = torch.jit.script(model.cpu()) # TorchScript 형식으로 내보내기
model_scripted.save('cifar10.pt')
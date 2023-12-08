#%%
import torch
import torch.optim 
import numpy as np
from tqdm import tqdm
# import copy
import wandb
import importlib
from data import load_semi_MNIST
import model_class as mod
importlib.reload(mod)

import torch
import torch.nn as nn
import torch.nn.functional as F


#%%
torch.manual_seed(1337)
np.random.seed(1337)

config = {'input_dim' : 28*28,
          'hidden_dim' : 500,
          'latent_dim' : 2,
          'batch_size' : 100,
          'labelled_size' : 3000,
          'epochs' : 2,
          'lr' : 0.0003,
          'best_loss' : 10**9,
          'patience_limit' : 3}
#%%
# wandb.init(project="VAE", config=config)
is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')
print('Current cuda device is', device)

#%%
labelled, unlabelled, validation = load_semi_MNIST(config['batch_size'], config['labelled_size'])


#%%
def kld(mu, logvar):
    kl = 0.5 * (mu**2 + logvar.exp() - logvar - 1)

    return torch.sum(kl, dim=-1)

def log_prior(p):
    prior = F.softmax(torch.ones_like(p), dim=-1)
    prior.requires_grad = False
    cross_entropy = -torch.sum(p * torch.log(prior), dim = -1)

    return cross_entropy    

def loss_function(x, x_reconst, mu, logvar, label):
    kl_div = kld(mu, logvar)
    reconst_loss = torch.sum(F.binary_cross_entropy(x_reconst, x, reduction='none'), dim = -1)
    prior = log_prior(label)
    L = kl_div + reconst_loss + prior
    
    return L

#%%
model = mod.VAE2(x_dim=config['input_dim'], h_dim = config['hidden_dim'], z_dim = config['latent_dim']).to(device)
# model = mod.DeepGenerativeModel([784, 10, 2, [600, 600]])
# optimizer = torch.optim.RMSprop(model.parameters(), lr = config['lr'], momentum=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))
# parameter 초기값 N(0, 0.01)에서 random sampling
# for param in model.parameters():
#     torch.nn.init.normal_(param, 0, 0.001)
#%% 

def onehot(digit):
    vector = torch.zeros(10)
    vector[digit] = 1
    return vector


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
        
        # labelled data loss
        x_reconst, mu, logvar = model(x, label)
        L = torch.mean(loss_function(x, x_reconst, mu, logvar, label))
        
        # unlabelled data loss
        u_prob = model.classify(u)
        temp_label = torch.cat([torch.nn.functional.one_hot(torch.zeros(len(u)).long() + i, num_classes=10) for i in range(10)], dim=0).float()
        extend_u = u.repeat(10, 1)

        u_reconst, u_mu, u_logvar = model(extend_u, temp_label)

        u_elbo = loss_function(extend_u, u_reconst, u_mu, u_logvar, temp_label)
        u_elbo = u_elbo.view_as(u_prob.t()).t()
        
        U = torch.sum(torch.mul(u_prob, u_elbo), dim = -1)
        H = torch.sum(torch.mul(u_prob, torch.log(u_prob + 1e-8)), dim = -1)
        
        J = L + torch.mean(U + H)

        #Classification loss
        prob = model.classify(x)
        # classification_loss = F.cross_entropy(prob, label, reduction='mean')*0.1*config['labelled_size']
        classification_loss = -torch.sum(label * torch.log(prob + 1e-8), dim=1).mean()*0.1*config['labelled_size']

        loss = J + classification_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        print(loss)
    print('Epoch: {} Train_Loss: {} :'.format(epoch, train_loss/config['batch_size']))    
    # wandb.log({'train_loss':train_loss/len(train_dataloader.dataset)})




#%%

x.shape
U.shape
uy.shape
label.shape
temp_label.shape

import torch
temp_label = torch.zeros([94,10])
p = temp_label
# 예제 행렬 생성
original_matrix = torch.tensor([[1, 2, 3],
                                [4, 5, 6]])

# 행렬을 복제하고 아래로 연결하기
replicated_matrix = original_matrix.repeat(3, 1)  # 3번 행 방향으로, 1번 열 방향으로 복제
result_matrix = torch.cat([original_matrix, replicated_matrix], dim=0)

print(result_matrix)


u = torch.zeros(5)  # 임의로 길이가 5인 텐서를 생성 (실제로는 데이터에 따라서 길이가 달라질 것입니다)

out = torch.stack([torch.nn.functional.one_hot(torch.zeros(len(u)).long() + i, num_classes=10) for i in range(10)], dim=0)

print(out)

'''
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_val, _ in test_dataloader:
            x_val = x_val.view(-1, img_size)
            x_val = x_val.to(device)
            x_val_reconst, mu, logvar = model(x_val)

            loss = mod.loss_func(x_val, x_val_reconst, mu, logvar).item()
            val_loss += loss/len(test_dataloader.dataset)
        val.append(val_loss)
        # wandb.log({'train_loss':train_loss/len(train_dataloader.dataset), 'valid_loss': val_loss})
        print(epoch, val_loss)
        if abs(val_loss - best_loss) < 1e-3: # loss가 개선되지 않은 경우
            patience_check += 1

            if patience_check >= patience_limit: # early stopping 조건 만족 시 조기 종료
                print("Learning End. Best_Loss:{:6f}".format(best_loss))
                break

        else: # loss가 개선된 경우
            best_loss = val_loss
            best_model = copy.deepcopy(model)
            patience_check = 0
'''
#%%
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
                              torch.FloatTensor(onehot(4))])).reshape(-1,28,28) 
                              for i in grid]
latent_grid_img = torchvision.utils.make_grid(latent_image, nrow=10)
plt.imshow(latent_grid_img.permute(1,2,0))
plt.show()
# wandb.log({"latent generate": wandb.Image(latent_grid_img)})
#%%

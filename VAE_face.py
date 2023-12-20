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
          'hidden_dim' : 500,
          'latent_dim' : 50,
          'batch_size' : 500,
          'labelled_size' : 50000,
          'epochs' : 200,
          'lr' : 0.0003,
          'best_loss' : 10**9,
          'patience_limit' : 3}

# set seed
torch.manual_seed(23)

#%%
wb_log = False
if wb_log: wandb.init(project="cifar10", config=config)
is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')
print('Current cuda device is', device)
#%%
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'],
                                          shuffle=True)


#%%
model = mod.VAE_norm(x_dim=config['input_dim'], h_dim = config['hidden_dim'], z_dim = config['latent_dim']).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))

#%%

img_size = config['input_dim']  
best_loss = config['best_loss']
patience_limit = config['patience_limit']
patience_check = 0 # 현재 몇 epoch 연속으로 loss 개선이 안되는지를 기록
val = []
for epoch in tqdm(range(config['epochs'])):
    model.train()
    train_loss = 0
    for x, _ in train_dataloader:
        # forward
        x = x.view(-1, img_size)
        x = x.to(device)
        x_reconst, mu_de, logvar_de, mu, logvar = model(x)

        # backprop and optimize
        loss = mod.loss_norm(x, mu_de, logvar_de, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    print('Epoch: {} Train_Loss: {} :'.format(epoch, train_loss/len(train_dataloader.dataset)))    
    
    
#%%
    
import torchvision.utils
import matplotlib.pyplot as plt
import numpy as np

model = torch.jit.load('vcifar10.pt')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 학습용 이미지를 무작위로 가져오기
dataiter = iter(train_dataloader)
images, labels = next(dataiter)

# 이미지 보여주기
imshow(torchvision.utils.make_grid(images[10]))
    

images[1].shape

x = images[10].view(-1, img_size).squeeze()

x_reconst, mu_de, logvar_de, mu, logvar = model(x)
imshow(x_reconst.reshape([3, 32, 32]).detach())

#%%
model.encoder(x)


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
                torch.nn.init.xavier_normal(m.weight.data)
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
def kld(mu, logvar):
    kl = 0.5 * (mu**2 + logvar.exp() - logvar - 1)

    return torch.sum(kl, dim=-1)

def log_prior(p):
    prior = F.softmax(torch.ones_like(p), dim=-1)
    prior.requires_grad = False
    cross_entropy = -torch.sum(p * torch.log(prior), dim = -1)

    return cross_entropy    

def loss_func2(x, x_reconst, mu, logvar, label):
    kl_div = kld(mu, logvar)
    reconst_loss = torch.sum(F.binary_cross_entropy(x_reconst, x, reduction='none'), dim = -1)
    prior = log_prior(label)
    L = kl_div + reconst_loss + prior
    
    return L

#%%
model = mod.VAE2(x_dim=config['input_dim'], h_dim = config['hidden_dim'], z_dim = config['latent_dim']).to(device)
# model = DeepGenerativeModel([784, 10, 2, [600, 600]])
# optimizer = torch.optim.RMSprop(model.parameters(), lr = config['lr'], momentum=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))
# parameter 초기값 N(0, 0.01)에서 random sampling
for param in model.parameters():
    torch.nn.init.normal_(param, 0, 0.001)
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
        L = torch.mean(loss_func2(x, x_reconst, mu, logvar, label))
        
        # unlabelled data loss
        u_prob = model.classifier(u)
        temp_label = torch.cat([torch.nn.functional.one_hot(torch.zeros(len(u)).long() + i, num_classes=10) for i in range(10)], dim=0).float()
        extend_u = torch.cat([u for _ in range(10)], dim=0)
        u_reconst, u_mu, u_logvar = model(extend_u, temp_label)
        u_elbo = loss_func2(extend_u, u_reconst, u_mu, u_logvar, temp_label)
        u_elbo = u_elbo.view_as(u_prob.t()).t()
        
        U = torch.sum(torch.mul(u_prob, u_elbo), dim = -1)
        H = torch.sum(torch.mul(u_prob, torch.log(u_prob + 1e-8)), dim = -1)
        
        J = L + torch.mean(U + H)

        #Classification loss
        prob = model.classifier(x)
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
                              torch.FloatTensor([0,0,1,0,0,0,0,0,0,0])])).reshape(-1,28,28) 
                              for i in grid]
latent_grid_img = torchvision.utils.make_grid(latent_image, nrow=10)
plt.imshow(latent_grid_img.permute(1,2,0))
plt.show()
# wandb.log({"latent generate": wandb.Image(latent_grid_img)})


#%%

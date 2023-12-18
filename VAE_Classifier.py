#%%
import torch
import torch.optim 
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import copy
import wandb
import importlib
from data import load_semi_MNIST
import model_class as mod
importlib.reload(mod)
#%%
config = {'input_dim' : 28*28,
          'hidden_dim' : 500,
          'latent_dim' : 50,
          'batch_size' : 500,
          'labelled_size' : 3000,
          'epochs' : 1000,
          'lr' : 0.0003,
          'best_loss' : 10**9,
          'patience_limit' : 50}

# set seed
seed = 423
torch.manual_seed(seed)
wb_log = True
#%%
if wb_log: wandb.init(project="ACC", name = f'M2/{config["labelled_size"]}', config=config)
is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')
print('Current cuda device is', device)
#%%
labelled, unlabelled, label_validation, unlabel_validation, test_loader = load_semi_MNIST(config['batch_size'],
                                                                                          config['labelled_size'],
                                                                                            seed_value=seed)

#%%
def kld(mu, logvar):
    kl = 0.5 * (mu**2 + logvar.exp() - logvar - 1)

    return torch.sum(kl, dim=-1)

def log_prior(p):
    prior = F.softmax(torch.ones_like(p), dim=-1)
    prior.requires_grad = False
    cross_entropy = -torch.sum(p * torch.log(prior), dim = -1)

    return cross_entropy    

def elbo(x, x_reconst, mu, logvar, label):
    kl_div = kld(mu, logvar)
    reconst_loss = torch.sum(F.binary_cross_entropy(x_reconst, x, reduction='none'), dim = -1)
    prior = log_prior(label)
    L = kl_div + reconst_loss + prior
    
    return L

def onehot(digit):
    vector = torch.zeros(10)
    vector[digit] = 1
    return vector

def loss_function(x, label, u, model):
    # labelled data loss
    x_reconst, mu, logvar = model(x, label)
    L = torch.mean(elbo(x, x_reconst, mu, logvar, label))
    
    # unlabelled data loss
    u_prob = model.classify(u)
    temp_label = torch.cat([F.one_hot(torch.zeros(len(u)).long() + i, num_classes=10) for i in range(10)], dim=0).float().to(device)
    extend_u = u.repeat(10, 1)

    u_reconst, u_mu, u_logvar = model(extend_u, temp_label)

    u_elbo = elbo(extend_u, u_reconst, u_mu, u_logvar, temp_label)
    u_elbo = u_elbo.view_as(u_prob.t()).t()
    
    U = torch.sum(torch.mul(u_prob, u_elbo), dim = -1)
    H = torch.sum(torch.mul(u_prob, torch.log(u_prob + 1e-8)), dim = -1)
    
    J = L + torch.mean(U + H)

    #Classification loss
    prob = model.classify(x)
    classification_loss = -torch.sum(label * torch.log(prob + 1e-8), dim=-1).mean()*0.1*config['labelled_size']

    loss = J + classification_loss
    return loss

#%%

model = mod.VAE2(x_dim=config['input_dim'], h_dim = config['hidden_dim'], z_dim = config['latent_dim']).to(device)
# optimizer = torch.optim.RMSprop(model.parameters(), lr = config['lr'], momentum=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))
# optimizer = torch.optim.RMSprop(model.parameters(), lr = 0.0003, alpha = 0.1, eps = 0.001, momentum = 0.9, centered = True)
# alpha : first momentum decay = 0.1
# eps : second momentum decay = 0.001
# momentum
# centered : initialization bisas correction


#%%
img_size = config['input_dim']  
best_acc = 0
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

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for (x, target), (u, _) in zip(label_validation, unlabel_validation):
            label = torch.stack([onehot(i) for i in target]).to(device)
            x = x.view(-1, img_size).to(device)
            u = u.view(-1, img_size).to(device)
    
            loss = loss_function(x, label, u, model)
            val_loss += loss/len(label_validation)
        val.append(val_loss)
        print(epoch, val_loss)

        # if abs(val_loss - best_loss) < 1: # loss가 개선되지 않은 경우
        #     patience_check += 1

        #     if patience_check >= patience_limit: # early stopping 조건 만족 시 조기 종료
        #         print("Learning End. Best_Loss:{:6f}".format(best_loss))
        #         break

        # else: # loss가 개선된 경우
        #     best_loss = val_loss
        #     best_model = copy.deepcopy(model)
        #     patience_check = 0
        
        accuracy = 0
        for x, label in test_loader:
            x = x.view(-1, img_size).to(device)
            pred_idx = torch.argmax(model.classify(x), dim=-1)
            accuracy += torch.mean((pred_idx.data.to(device) == label.to(device)).float())
        print(f'{accuracy.item()/len(test_loader)*100:.2f}%')
        if wb_log: wandb.log({'train_loss':train_loss/len(labelled), 'valid_loss': val_loss, 'Accuracy': accuracy.item()/len(test_loader)*100})

        if accuracy < best_acc: # loss가 개선되지 않은 경우
            patience_check += 1

            if patience_check >= patience_limit: # early stopping 조건 만족 시 조기 종료
                print("Learning End. Best_ACC:{:6f}".format(best_acc.item()/len(test_loader)*100))
                # break

        else: # loss가 개선된 경우
            best_acc = accuracy
            best_model = copy.deepcopy(model)
            patience_check = 0

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

with torch.no_grad():
    # manifold image
    if config['latent_dim'] == 2 :
        for j in range(10):
            latent_image = [best_model.decoder(torch.cat([torch.FloatTensor(i), 
                                        torch.FloatTensor(onehot(j))]).to(device)).reshape(-1,28,28) 
                                        for i in grid]
            latent_grid_img = torchvision.utils.make_grid(latent_image, nrow=10)
            if not wb_log: 
                plt.imshow(latent_grid_img.permute(1,2,0))
                plt.show()
            if wb_log: wandb.log({"latent generate": wandb.Image(latent_grid_img)})

    # accuracy
    best_model = best_model.to('cpu')
    accuracy = 0
    for x, label in test_loader:
        x = x.view(-1, img_size)
        pred_idx = torch.argmax(best_model.classify(x), dim=-1)
        accuracy += torch.mean((pred_idx.data == label).float())
    print(f'{accuracy.item()/len(test_loader)*100:.2f}%')
    if wb_log: wandb.log({'Best_Accuracy': accuracy.item()/len(test_loader)*100})  

    # analogies
    test_data = next(iter(test_loader))
    image = test_data[0][:10]
    test_label = test_data[1][:10]
    latent = [best_model.encoder(torch.cat([image[i].view(-1, img_size).squeeze(), torch.FloatTensor(onehot(test_label[i]))]))[0] for i in range(10)]
    analogies_image = []
    for i, z in enumerate(latent):
        gen_image = [best_model.decoder(torch.cat([z, torch.FloatTensor(onehot(j))])).reshape(-1,28,28) 
                                    for j in range(10)]
        gen_image.insert(0, image[i])
        analogies_image.extend(gen_image)
    gen_grid_img = torchvision.utils.make_grid(analogies_image, nrow=11)
    if not wb_log: plt.imshow(gen_grid_img.permute(1,2,0))
    if wb_log: wandb.log({"analogies generate": wandb.Image(gen_grid_img)})
# %%

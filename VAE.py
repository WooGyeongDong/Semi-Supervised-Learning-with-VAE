#%%
import torch
import torch.optim 
from torch.utils.data import DataLoader

from tqdm import tqdm
import copy
import wandb
import importlib
from data import load_MNIST
import model_class as mod
importlib.reload(mod)
#%%
config = {'input_dim' : 28*28,
          'hidden_dim' : 600,
          'latent_dim' : 50,
          'batch_size' : 100,
          'epochs' : 500,
          'lr' : 0.0003,
          'best_loss' : 10**9,
          'patience_limit' : 3}
#%%
# wandb.init(project="VAE_M1", config=config)
is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')
print('Current cuda device is', device)

#%%
train_data, test_data = load_MNIST()

train_dataloader = DataLoader(train_data, batch_size = config['batch_size'], shuffle=True )
test_dataloader = DataLoader(test_data, batch_size = config['batch_size'])

#%%
model = mod.VAE(x_dim=config['input_dim'], h_dim = config['hidden_dim'], z_dim = config['latent_dim']).to(device)
optimizer = torch.optim.RMSprop(model.parameters(), lr = config['lr'], momentum=0.1)

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
        x_reconst, mu, logvar = model(x)

        # backprop and optimize
        loss = mod.loss_func(x, x_reconst, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print('Epoch: {} Train_Loss: {} :'.format(epoch, train_loss/len(train_dataloader.dataset)))    
    
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
        if abs(val_loss - best_loss) < 0.05: # loss가 개선되지 않은 경우
            patience_check += 1

            if patience_check >= patience_limit: # early stopping 조건 만족 시 조기 종료
                print("Learning End. Best_Loss:{:6f}".format(best_loss))
                break

        else: # loss가 개선된 경우
            best_loss = val_loss
            best_model = copy.deepcopy(model)
            patience_check = 0

#%%
# import torchvision.utils

# image = [test_data[i][0] for i in range(9)]
# grid_img = torchvision.utils.make_grid(image, nrow=3)

# generate_image = [model(image[i].view(-1, img_size).to(device))[0].reshape(-1,28,28) for i in range(9)]
# gen_grid_img = torchvision.utils.make_grid(generate_image, nrow=3)

# grid = []
# for i in range(20):
#     for j in range(20):
#         grid.append(((i/3-3),(j/3-3)))
# latent_image = [model.decoder(torch.tensor(i).to(device)).reshape(-1,28,28) for i in grid]
# latent_grid_img = torchvision.utils.make_grid(latent_image, nrow=20)


# wandb.log({"original": wandb.Image(grid_img),
#            "generate": wandb.Image(gen_grid_img),
#            "latent generate": wandb.Image(latent_grid_img)})


#%%

model_scripted = torch.jit.script(best_model.cpu()) # TorchScript 형식으로 내보내기
model_scripted.save('M1.pt')


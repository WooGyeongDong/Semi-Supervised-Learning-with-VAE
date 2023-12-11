import torch
from torchvision.datasets import MNIST 
from torchvision import transforms
from scipy import io
from torch.utils.data import DataLoader, SubsetRandomSampler
import random

def load_MNIST() :
    train_data = MNIST(root = './data/02/',
                                train=True,
                                download=True,
                                transform=transforms.ToTensor())
    test_data = MNIST(root = './data/02/',
                                train=False,
                                download=True,
                                transform=transforms.ToTensor())
    print('number of training data : ', len(train_data))
    print('number of test data : ', len(test_data))
    
    return train_data, test_data

def load_frey_face() :
    file = io.loadmat("frey_rawface.mat")
    file = file["ff"].T.reshape(-1,28,20)
    file = torch.from_numpy(file).float()/255
    
    return file 

def load_semi_MNIST(batch_size, labelled_size, seed_value = 23):
    random.seed(seed_value)
    train_data, test_data = load_MNIST()
    
    indices = list(range(len(train_data)))
    random.shuffle(indices)
    label_valid_indices = indices[:int(labelled_size/5)]
    unlabel_valid_indices = indices[int(labelled_size/5):10000]
    labelled_indices = indices[10000:10000+labelled_size]
    unlabelled_indices = indices[10000+labelled_size:]

    labelled_batch_size = int(labelled_size*batch_size/50000)

    labelled = DataLoader(train_data, batch_size=labelled_batch_size, pin_memory=True,
                                            sampler=SubsetRandomSampler(labelled_indices))
    unlabelled = DataLoader(train_data, batch_size=batch_size-labelled_batch_size, pin_memory=True,
                                                sampler=SubsetRandomSampler(unlabelled_indices))
    label_validation = DataLoader(train_data, batch_size=batch_size, pin_memory=True,
                                                sampler=SubsetRandomSampler(label_valid_indices))
    unlabel_validation = DataLoader(train_data, batch_size=labelled_batch_size, pin_memory=True,
                                                sampler=SubsetRandomSampler(unlabel_valid_indices))
    return labelled, unlabelled, label_validation, unlabel_validation


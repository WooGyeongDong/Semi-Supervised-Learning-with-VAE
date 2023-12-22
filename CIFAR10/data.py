import torch
from torchvision.datasets import MNIST 
from torchvision import transforms
from scipy import io
from torch.utils.data import DataLoader, SubsetRandomSampler
import random


from torchvision.datasets import CIFAR10

def load_CIFAR10(batch_size, labelled_size, seed_value = 23):
    random.seed(seed_value)
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    indices = list(range(len(trainset)))
    random.shuffle(indices)
    labelled_indices = indices[:labelled_size]
    unlabelled_indices = indices[labelled_size:]
    labelled_batch_size = int(labelled_size*batch_size/50000)
    labelled = DataLoader(trainset, batch_size=labelled_batch_size, pin_memory=True,
                                            sampler=SubsetRandomSampler(labelled_indices))
    unlabelled = DataLoader(trainset, batch_size=batch_size-labelled_batch_size, pin_memory=True,
                                                sampler=SubsetRandomSampler(unlabelled_indices))
    
    testset = CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return labelled, unlabelled, test_loader

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from hmdb51_dataset import HMDB51

def trans_S():
    return transforms.Compose([transforms.Resize(255),
                 transforms.RandomCrop(224),
                 transforms.RandomHorizontalFlip(0.5),
                 transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.1),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))])

def trans_T():
    return transforms.Compose([transforms.Resize(255),
                 transforms.RandomCrop(224),
                 transforms.RandomHorizontalFlip(0.5),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))])

def dataloader_S(minibatch_size, train):
    hmdb51_S = HMDB51(transform=trans_S(),
                    train=train,
                    Spatial=True)
    dataloader_S = DataLoader(dataset=hmdb51_S,
                            batch_size=minibatch_size,
                            num_workers=4,
                            shuffle=True)
    return dataloader_S

def dataloader_T(minibatch_size, train):
    hmdb51_T = HMDB51(transform=trans_T(),
                        train=train,
                        Spatial=False)
    dataloader_T= DataLoader(dataset=hmdb51_T,
                            batch_size=minibatch_size,
                            num_workers=4,
                            shuffle=True)
    return dataloader_T

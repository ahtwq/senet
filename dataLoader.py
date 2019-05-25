import os
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


def default_loader(img):
    return Image.open(img)
    
class Mydataset(Dataset):
    def __init__(self, img_root, txtfile, img_transform=None, label_transform=None, loader=default_loader):
        with open(txtfile, 'r') as f:
            lines = f.readlines()
        self.img_list = [os.path.join(img_root, i.split()[0]+'.png') for i in lines if os.path.exists(os.path.join(img_root, i.split()[0]+'.png'))]
        self.label_list = [int(i.split()[1]) for i in lines if os.path.exists(os.path.join(img_root,i.split()[0]+'.png'))]
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.loader = loader

    def __getitem__(self, index):
        img_path = self.img_list[index]
        label = (self.label_list[index])
        
        img = self.loader(img_path)
        if self.img_transform is not None:
            img = self.img_transform(img)
        return img, label

    def __len__(self):
        return len(self.label_list)

                     
def loader(img_root, train_txt, test_txt, Train_batchSize, Test_batchSize):

    train_transform = transforms.Compose([
                            transforms.Resize(512*2),
                            transforms.RandomCrop(448*2),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
                            
    val_transform =  transforms.Compose([
                         transforms.Resize(512*2),
                         transforms.CenterCrop(448*2),
                         transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                         ])

    train_set = Mydataset(img_root=img_root, txtfile=train_txt, img_transform=train_transform)                       
    test_set = Mydataset(img_root=img_root, txtfile=test_txt, img_transform=val_transform)

    loaders = {
        'train': DataLoader(
            train_set,
            batch_size=Train_batchSize,
            shuffle=True,
            num_workers=4,
        ),
        'test': DataLoader(
            test_set,
            batch_size=Test_batchSize,
            shuffle=False,
            num_workers=4,
        )
    }
    return loaders

def load_cifar10(Train_batchSize=32, Test_batchSize=16):
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    
    loaders = {
        'train': DataLoader(
            train_set,
            batch_size=Train_batchSize,
            shuffle=True,
            num_workers=4,
        ),
        'test': DataLoader(
            test_set,
            batch_size=Test_batchSize,
            shuffle=False,
            num_workers=4,
        )
    }
    return loaders

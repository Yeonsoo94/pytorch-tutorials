import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
class DataLoader:

    classes_mnist = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10')
    classes_CIFAR = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def __init__(self, valid_size =.2, batch_size = 4, imageSize = 0, imageType = 'CIFAR'):
        super(DataLoader, self).__init__()

        self.valid_size = valid_size
        self.batch_size = batch_size
        self.imageSize = imageSize
        self.imageType = imageType

    def setdata(self):

        if self.imageSize == 0:
            transform = transforms.Compose(
                 [
                 transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(self.imageSize),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)

        return trainset, testset

    def splitDataset(self, dataset):

        lenOfdata = len(dataset)
        indices = list(range(lenOfdata))
        split = int(np.floor(self.valid_size * lenOfdata))
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        return train_sampler, valid_sampler

    def loadDataset(self, train_sampler, valid_sampler, trainset, testset):
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=2)

        validloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size,
            sampler=valid_sampler,
            num_workers=2)

        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=2)

        return trainloader, validloader, testloader


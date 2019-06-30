import torch
import torchvision
import torchvision.transforms as transforms

class DataLoader:

    classes_mnist = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10')
    classes_CIFAR = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def __init__(self, valid_size =.2, batch_size = 4, imageType = 'CIFAR'):
        super(DataLoader, self).__init__()

        self.valid_size = valid_size
        self.batch_size = batch_size

        trainset, testset = setdata()
        train_sampler,valid_sampler = splitDataset()
        trainloader, validloader, testloader = loadDataset(train_sampler, valid_sampler)

    def setdata(self):

        transform = transforms.Compose(
             [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)

    def splitDataset(self, dataset, valid_size):

        lenOfdata = len(dataset)
        indices = list(range(lenOfdata))
        split = int(np.floor(valid_size * lenOfdata))
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        return train_sampler,valid_sampler

    def loadDataset(self, batch_size, train_sampler, valid_sampler):
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size,
            sampler=train_sampler,
            num_workers=2)

        validloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size,
            sampler=valid_sampler,
            num_workers=2)

        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

        return trainloader, validloader, testloader
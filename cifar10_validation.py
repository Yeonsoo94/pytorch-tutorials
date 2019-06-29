import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

valid_size = .2
batch_size = 8

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

num_train = len(trainset)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))
np.random.shuffle(indices)


train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

trainloader = torch.utils.data.DataLoader(
                                          trainset, batch_size=batch_size,
                                          sampler=train_sampler,
                                          num_workers=2)

validloader = torch.utils.data.DataLoader(
                                          trainset, batch_size=batch_size,
                                          sampler=valid_sampler,
                                          num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

epochs = 15
mini_batch = 1000
mini_batch_1 = 999

train_accuracy, test_accuracy = [], []
train_losses, test_losses = [], []
label_acc_per_epoch = [[0] * epochs for i in range(10)]

for epoch in range(epochs):  # loop over the dataset multiple times

    total_train = 0.0
    valid_accuracy = 0.0
    train_acc = 0.0

    Iterate_accuracy, Iterate_losses = 0.0, 0.0
    Iterater = 0;
    running_loss = 0.0

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    for i, (data, target) in enumerate(trainloader):  ##  num Of train Images / batch Size
        correct_train_iter = 0.0
        total_train_iter = 0.0

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        # accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_train_iter += target.size(0)  ##32
        correct_train_iter += predicted.eq(target).sum().item()  ## 0~32

        if i % mini_batch == mini_batch_1:  # print every 2000 mini-batches
            train_acc_iter = correct_train_iter / total_train_iter
            train_loss_iter = running_loss / mini_batch

            print('\n[%d, %5d] loss: %.3f accuracy: %.3f' % (epoch + 1, i + 1, train_loss_iter, train_acc_iter))

            Iterate_accuracy += train_acc_iter
            Iterate_losses += train_loss_iter
            Iterater += 1;

            running_loss = 0.0

    train_accuracy.append(Iterate_accuracy / Iterater)
    train_losses.append(Iterate_losses / Iterater)

    net.eval()
    with torch.no_grad():
        step = 0
        valid_loss = 0
        valid_accuracy = 0

        for inputs, labels in validloader:  ##  len(validloader) num Of valid Images(16000) / batch Size (16) = 1000

            ##Logic 1
            logps = net.forward(inputs)
            batch_loss = criterion(logps, labels)
            valid_loss += batch_loss.item()
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = (top_class == labels.view(*top_class.shape))
            # print(equals)

            ## Logic 2
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            # print(c)

            valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            step += 1

            for i in range(10):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

        test_accuracy.append(valid_accuracy / step)
        test_losses.append(valid_loss / step)

    net.train()

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
        label_acc_per_epoch[i][epoch] = (100 * class_correct[i] / class_total[i])

    print("Epoch %d / %d .. " % (epoch + 1, epochs))
    print(train_accuracy[epoch])
    print(train_losses[epoch])
    print(test_accuracy[epoch])
    print(test_losses[epoch])

    # print(f"***Epoch {epoch+1}/{epochs}.. "
    # f"Train loss: {running_loss / len(trainloader):.3f}.. "
    # f"Train accuracy: {correct_train / total_train:.3f}.. "
    # f"Validation loss: {valid_loss/len(validloader):.3f}.. "
    # f"Validation accuracy: {valid_accuracy/len(validloader):.3f}")

print('Finished Training')

###draw Result
fig = plt.figure()
fig.set_size_inches(10.5, 7.5)

ax11 = fig.add_subplot(2, 2, 1)
ax11.set_xlim([0, 10])
ax11.set_ylim([0.94, 1.001])
ax11.set_title('Training Accuracy')
ax11.margins(x=0.1, y=0.05)

ax12 = fig.add_subplot(2, 2, 2)
ax12.set_xlim([0, 10])
ax12.set_ylim([0.0, 0.5])
ax12.set_title('Training Loss')
ax12.margins(x=0.1, y=0.5)

ax21 = fig.add_subplot(2, 2, 3)
ax21.set_xlim([0, 10])
ax21.set_ylim([0.97, 1.001])
ax21.set_title('Validation Accuracy')

ax22 = fig.add_subplot(2, 2, 4)
ax22.set_xlim([1, 10])
ax22.set_ylim([0.0, 0.1])
ax22.set_title('Validation Loss')

print(running_loss)

ax11.plot(train_accuracy)
ax12.plot(train_losses)
ax21.plot(test_accuracy)
ax22.plot(test_losses)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

outputs = net(images)

def drawResult():

    fig = plt.figure()
    fig.set_size_inches(10.5, 7.5)
    plot_title = ["1", "2", "3", "4"]

    subplots = []

    for i in range(1, 4):
        subplot = fig.add_subplot(2, 2, i)
        subplot.set_xlim([0, 10])
        subplot.set_ylim([0.94, 1.001])
        subplot.set_title(plot_title[i])
        subplot.margins(x=0.1, y=0.05)
        subplots[0] = subplot

    subplots[0].plot(train_accuracy)
    subplots[1].plot(train_losses)
    subplots[2].plot(test_accuracy)
    subplots[3].plot(test_losses)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torchvision import models
import logging
from datetime import datetime
from pytz import timezone
import json
import os

batch_size = 16

transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(299),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np

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


# net = Net()

net = models.vgg19(pretrained='true')
net.classifier[6] = nn.Linear(4096, 10)

# net = models.inception_v3(pretrained=True, aux_logits=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.warning(device)
net.to(device)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


def validate(validloader, net):
    running_loss = 0;
    step = 0;

    for inputs, labels in validloader:  ##  len(validloader) num Of valid Images(16000) / batch Size (16) = 1000

        step += 1

        outputs = net(inputs.to(device))
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels.to(device)).squeeze()

        loss = criterion(outputs, labels.to(device))
        running_loss += loss.item()

        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

    return class_correct, class_total, (running_loss / len(validloader))


######
start_time = str(datetime.now(timezone('US/Pacific')).strftime('%m-%d_%H:%M:%S'))
file_name = start_time + 'log.txt'
w = open(file_name, 'a+')
#####


epochs = 30
mini_batch = 500
mini_batch_1 = mini_batch - 1
w.write('\nepochs : {}\n\n'.format(epochs))
w.write('\nmini_batch : {}\n\n'.format(mini_batch))
w.write('\n_batch_size : {}\n\n'.format(batch_size))
w.write('\nmodel : {}\n\n'.format('VGG19'))

train_losses = list(0. for i in range(epochs))
test_losses = list(0. for i in range(epochs))

train_accuracies = list(0. for i in range(epochs))
test_accuracies = list(0. for i in range(epochs))

label_acc_per_epoch = [[0] * epochs for i in range(10)]
label_val_per_epoch = [[0] * epochs for i in range(10)]

running_loss = 0.0
for epoch in range(epochs):  # loop over the dataset multiple times

    epoch_loss = 0.0
    w.write('\n\epoch : {}\n\n'.format(epoch))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    class_correct_train = list(0. for i in range(10))
    class_total_train = list(0. for i in range(10))

    for i, (data, target) in enumerate(trainloader, 0):  ##  num Of train Images / batch Size

        correct_train_iter, total_train_iter = 0.0, 0.0

        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(data.to(device))
        loss = criterion(outputs, target.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        epoch_loss += loss.item()

        # accuracy 1
        _, predicted = torch.max(outputs.data, 1)
        total_train_iter += target.size(0)  ##32
        correct_train_iter += predicted.eq(target.to(device)).sum().item()  ## 0~32

        # accuracy 2
        _, predicted2 = torch.max(outputs, 1)
        c = (predicted == target.to(device)).squeeze()

        for j in range(len(target)):
            label = target[j]
            class_correct_train[label] += c[j].item()
            class_total_train[label] += 1

        if i % mini_batch == mini_batch_1:  # print every 2000 mini-batches
            train_acc_iter = correct_train_iter / total_train_iter
            train_loss_iter = running_loss / mini_batch

            print('\n[%d, %5d] loss: %.3f accuracy: %.3f' % (epoch + 1, i + 1, train_loss_iter, train_acc_iter))
            logging.warning(
                '\n[%d, %5d] loss: %.3f accuracy: %.3f' % (epoch + 1, i + 1, train_loss_iter, train_acc_iter))

            running_loss = 0.0

    ## finish 1 epoch ##
    train_losses[epoch] = (epoch_loss / len(trainloader))
    print(train_losses)
    print(len(trainloader))

    train_accuracy = 0
    for i in range(10):
        acc = class_correct_train[i] / class_total_train[i]
        label_acc_per_epoch[i][epoch] = acc
        train_accuracy += acc
        print('Accuracy of Train %5s : %.3f %%' % (classes[i], acc))
    train_accuracies[epoch] = train_accuracy / 10

    with torch.no_grad():
        class_correct, class_total, running_loss = validate(testloader, net)
        test_losses[epoch] = running_loss

    valid_accuracy = 0
    for i in range(10):
        acc = (class_correct[i] / class_total[i])
        valid_accuracy += acc
        print('Accuracy of Validation %5s : %.3f %%' % (classes[i], acc))
        label_val_per_epoch[i][epoch] = acc
    test_accuracies[epoch] = valid_accuracy / 10

    print("Epoch %d / %d .. " % (epoch + 1, epochs))
    print(train_accuracies[epoch])
    print(train_losses[epoch])
    print(test_accuracies[epoch])
    print(test_losses[epoch])

    logging.warning("Epoch %d / %d .. " % (epoch + 1, epochs))
    logging.warning('\n train_accuracy: %.3f' % (train_accuracies[epoch]))
    logging.warning('\n train_losses: %.3f' % (train_losses[epoch]))
    logging.warning('\n test_accuracy: %.3f' % (test_accuracies[epoch]))
    logging.warning('\n test_losses: %.3f' % (test_losses[epoch]))

    # print(f"***Epoch {epoch+1}/{epochs}.. "
    # f"Train loss: {running_loss / len(trainloader):.3f}.. "
    # f"Train accuracy: {correct_train / total_train:.3f}.. "
    # f"Validation loss: {valid_loss/len(validloader):.3f}.. "
    # f"Validation accuracy: {valid_accuracy/len(validloader):.3f}")

    w.write("\n training accuracy")
    w.write(json.dumps(train_accuracies))
    w.write("\n train_losses")
    w.write(json.dumps(train_losses))
    w.write("\n test_accuracies")
    w.write(json.dumps(test_accuracies))
    w.write("\n test_losses")
    w.write(json.dumps(test_losses))

print('Finished Training')

for i in range(0, 10):
    epoch_list = label_val_per_epoch[i]
    label = '%s' % classes[i]

w.write("label_val_per_epoch")
w.write(json.dumps(label_val_per_epoch))
w.write("label_acc_per_epoch")
w.write(json.dumps(label_acc_per_epoch))

w.close()

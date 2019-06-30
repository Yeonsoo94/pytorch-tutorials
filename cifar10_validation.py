import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import net

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main(config):
    epochs = 15
    mini_batch = 1000
    mini_batch_1 = mini_batch - 1

    train_losses = list(0. for i in range(10))
    test_losses = list(0. for i in range(10))

    label_acc_per_epoch = [[0] * epochs for i in range(10)]
    label_val_per_epoch = [[0] * epochs for i in range(10)]

    DataLoader = DataLoader();

    # show Images and Labels per mini-batch
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

    net = net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):  # loop over the dataset multiple times

        epoch_loss = 0.0

        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))

        class_correct_train = list(0. for i in range(10))
        class_total_train = list(0. for i in range(10))

        for i, (data, target) in enumerate(trainloader, 0):  ##  num Of train Images / batch Size
            correct_train_iter, total_train_iter = 0.0, 0.0

            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            # accuracy 1
            _, predicted = torch.max(outputs.data, 1)
            total_train_iter += target.size(0)  ##32
            correct_train_iter += predicted.eq(target).sum().item()  ## 0~32

            # accuracy 2
            _, predicted2 = torch.max(outputs, 1)
            c = (predicted == target).squeeze()

            for j in range(batch_size):
                label = target[j]
                class_correct_train[label] += c[j].item()
                class_total_train[label] += 1

            if i % mini_batch == mini_batch_1:  # print every 2000 mini-batches
                train_acc_iter = correct_train_iter / total_train_iter
                train_loss_iter = running_loss / mini_batch

                epoch_loss += running_loss

                print('\n[%d, %5d] loss: %.3f accuracy: %.3f' % (epoch + 1, i + 1, train_loss_iter, train_acc_iter))

                running_loss = 0.0

        ## finish 1 epoch ##
        train_losses.append(epoch_loss / len(trainloader))

        train_accuracy = 0
        for i in range(10):
            acc = class_correct_train[i] / class_total_train[i]
            label_acc_per_epoch[i][epoch] = acc
            train_accuracy += acc
            print('Accuracy of Train %5s : %.3f %%' % (classes[i], acc))

        with torch.no_grad():
            class_correct, class_total, running_loss = validate(validloader, net)
            test_losses.append(running_loss)
            print(running_loss)
            print(test_losses[epoch])

        valid_accuracy = 0
        for i in range(10):
            acc = (class_correct[i] / class_total[i])
            valid_accuracy += acc
            print('Accuracy of Validation %5s : %3f %%' % (classes[i], acc))
            label_val_per_epoch[i][epoch] = acc

        print("Epoch %d / %d .. " % (epoch + 1, epochs))
        print(train_accuracy / 10)
        print(train_losses[epoch - 1])
        print(valid_accuracy / 10)
        print(test_losses[epoch - 1])


    print('Finished Training')

def validate(validloader, net):
    running_loss = 0;
    step = 0;

    for inputs, labels in validloader:  ##  len(validloader) num Of valid Images(16000) / batch Size (16) = 1000

        step += 1

        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()

        loss = criterion(outputs, labels)
        running_loss += loss.item()

        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

    return class_correct, class_total, (running_loss / step)



if __name__ == '__main__':

    testconfig = 'testnet_config.json'
    config = json.load(testconfig)

    main(config)
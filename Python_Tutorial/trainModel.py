from __future__ import print_function
from __future__ import division
import torch

import copy
import time


def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    since = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_losses = [0. for i in range(num_epochs)]
    test_losses = [0. for i in range(num_epochs)]

    train_accuracies = [0. for i in range(num_epochs)]
    test_accuracies = [0. for i in range(num_epochs)]

    label_acc_per_epoch = [[0] * num_epochs for i in range(10)]
    label_val_per_epoch = [[0] * num_epochs for i in range(10)]

    batch_loss = 0.0

    checkPoint = 1000;
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        epoch_loss = 0.0

        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))

        class_correct_train = list(0. for i in range(10))
        class_total_train = list(0. for i in range(10))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for j, (inputs, labels) in enumerate(dataloaders[phase]):

                inputs = inputs.to(device)
                labels = labels.to(device)

                correct_train_iter, total_train_iter = 0.0, 0.0

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    batch_loss += loss.item()

                    _, preds = torch.max(outputs, 1)
                    c = (preds == labels).squeeze()
                    total_train_iter += labels.size(0)  ##32
                    correct_train_iter += preds.eq(labels).sum().item()  ## 0~32

                    for i in range(len(labels)):
                        label = labels[i]
                        class_correct[label] += c[i].item()
                        class_total[label] += 1

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    if phase == 'train':
                        if j % checkPoint == checkPoint - 1:
                            train_acc_iter = correct_train_iter / total_train_iter
                            train_loss_iter = batch_loss / checkPoint

                            print('\n[%d, %5d] loss: %.3f accuracy: %.3f' % (
                                epoch + 1, j + 1, train_loss_iter, train_acc_iter))

                            batch_loss = 0.0

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # print(dataloaders[phase].dataset)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if phase == 'train':
                for i in range(10):
                    acc = class_correct[i] / class_total[i]
                    label_acc_per_epoch[i][epoch] = acc

                train_losses[epoch] = epoch_loss
                train_accuracies[epoch] = epoch_acc
            else:
                for i in range(10):
                    acc = class_correct[i] / class_total[i]
                    label_val_per_epoch[i][epoch] = acc

                test_losses[epoch] = epoch_loss
                test_accuracies[epoch] = epoch_acc

            print('{} Loss: {:.3f} Acc: {:.3f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:3f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    history = {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "test_losses": train_losses,
        "test_accuracies": train_accuracies,
        "label_acc_per_epoch": label_acc_per_epoch,
        "label_val_per_epoch": label_val_per_epoch
    }

    return model, history

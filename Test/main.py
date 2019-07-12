import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from Test import trainModel, dataLoader, InitializeModels
from datetime import datetime
from pytz import timezone

################
start_time = str(datetime.now(timezone('US/Pacific')).strftime('%m-%d_%H:%M:%S'))
file_name = start_time + 'log.txt'
w = open(file_name, 'a+')
################

num_epochs = 30
mini_batch = 300
mini_batch_1 = mini_batch - 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_name = 'net'
num_classes = 10

# Initialize the non-pretrained version of the model used for this run
scratch_model, input_size = InitializeModels.initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=False)

print(scratch_model)

scratch_model = scratch_model.to(device)
scratch_optimizer = optim.SGD(scratch_model.parameters(), lr=0.001, momentum=0.9)
scratch_criterion = nn.CrossEntropyLoss()

dataloader = dataLoader.DataLoader(valid_size =.2, batch_size = 4, imageSize = 0, imageType ='CIFAR')
trainset, testset = dataloader.setdata()
train_sampler, valid_sampler = dataloader.splitDataset(trainset)
trainloader, validloader, testloader = dataloader.loadDataset(train_sampler, valid_sampler, trainset, testset)
data_set = {'train': trainloader, 'val': validloader}
model, history = trainModel.train_model(scratch_model, data_set, scratch_criterion, scratch_optimizer, num_epochs=num_epochs, is_inception=(model_name == "inceptionX"))

###draw Result

fig = plt.figure()
fig.set_size_inches(10.5, 7.5)
plot_title = ["train_accuracies", "train_losses", "test_accuracies", "test_losses"]

for i in range(4):
    subplot = fig.add_subplot(2, 2, i+1)
    subplot.set_xlim([-0.2, num_epochs])
    subplot.set_ylim([0.0, 1 if i % 2 == 0 else max(history[plot_title[i]])+1])
    subplot.set_title(plot_title[i])
    subplot.plot(history[plot_title[i]], color = ("red" if i % 2 == 0 else "blue"))

plt.legend(frameon=False)
plt.tight_layout()
plt.show()


fig = plt.figure(1,)
fig.set_size_inches(15, 5.5)

for i in range(0, 10):
    a = plt.plot(history.label_val_per_epoch[i], label='%s' % dataLoader.classes_mnist[i])
    #a = plt.plot(history.label_acc_per_epoch[i], label='%s' % dataLoader.classes_mnist[i])

plt.title("validation")
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid(True)
plt.ylim((0.4,1))
plt.xlim((-0.3, num_epochs))
plt.tight_layout()
plt.legend(loc='lower center', ncol=5, frameon=False)
plt.show()

### tensorbord X
# Google tensorboard X for Pytorch
# Dweep suppose to review my code ...



import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import logging
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import InitializeModels
import trainModel
import dataLoader
from torchvision import models
from datetime import datetime
from pytz import timezone

################
start_time = str(datetime.now(timezone('US/Pacific')).strftime('%m-%d_%H:%M:%S'))
file_name = start_time + 'log.txt'
w = open(file_name, 'a+')
################

num_epochs = 30
mini_batch = 500
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

dataloader = dataLoader.DataLoader(valid_size =.2, batch_size = 4, imageSize = 0, imageType = 'CIFAR')
trainset, testset = dataloader.setdata()
train_sampler, valid_sampler = dataloader.splitDataset(trainset)
trainloader, validloader, testloader = dataloader.loadDataset(train_sampler, valid_sampler, trainset, testset)
data_set = {'train': trainloader, 'val': validloader}
_,scratch_hist = trainModel.train_model(scratch_model, data_set, scratch_criterion, scratch_optimizer, num_epochs=num_epochs, is_inception=(model_name=="inceptionX"))

# Plot the training curves of validation accuracy vs. number
#  of training epochs for the transfer learning method and
#  the model trained from scratch
ohist = []
shist = []

ohist = [h.cpu().numpy() for h in hist]
shist = [h.cpu().numpy() for h in scratch_hist]

plt.title("Validation Accuracy vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Validation Accuracy")
plt.plot(range(1,num_epochs+1),ohist,label="Pretrained")
plt.plot(range(1,num_epochs+1),shist,label="Scratch")
plt.ylim((0,1.))
plt.xticks(np.arange(1, num_epochs+1, 1.0))
plt.legend()
plt.show()

logging.warning(device)
net.to(device)





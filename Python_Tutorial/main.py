import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from Python_Tutorial import trainModel, dataLoader, initializeModels

num_epochs = 30
batchSize = 8
inputSize = 32 # 299 for VGG, 32 for CNN
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

models = ['alexnet', 'inceptionv3', 'alexnet', 'net']
model_name = models[3]

dataloader = dataLoader.DataLoader(batch_size = batchSize, imageSize = inputSize, imageType ='CIFAR')
trainset, testset = dataloader.setdata()

data_set = {'train': trainset, 'val': testset}


# Initialize the non-pretrained version of the model used for this run
scratch_model, input_size = initializeModels.initialize_model(model_name, num_classes=10, feature_extract=False, use_pretrained=False)

print(scratch_model)

scratch_model = scratch_model.to(device)
scratch_optimizer = optim.SGD(scratch_model.parameters(), lr=0.001, momentum=0.9)
scratch_criterion = nn.CrossEntropyLoss()

model, history = trainModel.train_model(scratch_model, data_set, scratch_criterion, scratch_optimizer, num_epochs=num_epochs)

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


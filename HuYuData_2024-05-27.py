import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import torch.optim as optim
import matplotlib.style as style
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
import time
from torch.autograd import Variable

warnings.filterwarnings('ignore')


batch_size = 4
learning_rate = 0.0001
epoch_range = 15
weight_decay = 0
dropout = True
xavier = False

trainLoss = []
testLoss = []
trainAccuracy = []
trainEpoch = []
testAccuracy = []


transform_cifar = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

traindataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_cifar)
train_set = torch.utils.data.DataLoader(traindataset, batch_size=batch_size,
                                        shuffle=True, num_workers=2)

testdataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_cifar)
test_set = torch.utils.data.DataLoader(testdataset, batch_size=batch_size,
                                       shuffle=False, num_workers=2)

labels = ('plane', 'car', 'bird', 'cat',
          'deer', 'dog', 'frog', 'horse', 'ship', 'truck')




class modelCIFAR(nn.Module):
    def __init__(self):
        super(modelCIFAR, self).__init__()
        self.convLayer1 = nn.Conv2d(3, 32, 5)  # First conv layer (3(input), 32(output), 5(filter_size))
        self.maxPool = nn.MaxPool2d(2, 2)  # Max Pool (2(filter_size), 2(stride))
        self.convLayer2 = nn.Conv2d(32, 64, 5)  # Second conv layer (32(input), 64(output), 5(filter_size))
        self.drop1 = nn.Dropout(0.2, inplace=False)  # Dropout layer with probability 0.2
        self.fullyc1 = nn.Linear(1600, 200)  # Fully Connected Layer (64*5*5(input), 200(output))
        self.xav = nn.init.xavier_normal_(self.fullyc1.weight)
        self.fullyc2 = nn.Linear(200, 100)  # Fully Connected Layer (200(input), 100(output))
        self.fullyc3 = nn.Linear(100, 10)  # Fully Connected Layer (100(input), 10(output))

    def forward(self, img):
        img = self.maxPool(self.drop1(F.relu(self.convLayer1(img))))
        img = self.maxPool(self.drop1(F.relu(self.convLayer2(img))))
        # img = self.maxPool(F.relu(self.convLayer1(img)))
        # img = self.maxPool(F.relu(self.convLayer2(img)))
        img = img.view(-1, 64 * 5 * 5)
        img = F.relu(self.fullyc1(img))
        img = F.relu(self.fullyc2(img))
        img = self.fullyc3(img)
        return img
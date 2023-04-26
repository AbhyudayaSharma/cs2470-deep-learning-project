# import the necessary packages
import torch
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import LeakyReLU
from torch.nn import LogSoftmax
from torch import flatten
from torch.nn import NLLLoss


class SimpleConvModel(Module):
    def __init__(self, numChannels, classes):
        super(SimpleConvModel, self).__init__()

        self.conv_1 = Conv2d(in_channels=numChannels, out_channels=20, kernel_size=(5, 5))
        self.relu_1 = LeakyReLU()
        self.max_pool_1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv_2 = Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))
        self.relu_2 = LeakyReLU()
        self.max_pool_2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc_1 = Linear(in_features=800, out_features=500)
        self.relu_3 = LeakyReLU()

        self.fc_2 = Linear(in_features=500, out_features=classes)
        self.log_softmax = LogSoftmax(dim=1)

        self._loss = NLLLoss()

    def forward(self, x):
        # convert to float16
        x = x.to(torch.float16)

        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.max_pool_1(x)

        x = self.conv_2(x)
        sx = self.relu_2(x)
        x = self.max_pool_2(x)

        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)
        output = self.log_softmax(x)
        return output

    def loss(self, logits, labels):
        return self._loss(logits, labels)

    def accuracy(self, logits, labels):
        correct = (logits == labels).float().sum()
        return 100 * correct / len(logits)

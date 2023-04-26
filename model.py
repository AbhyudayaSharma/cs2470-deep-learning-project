# import the necessary packages
import torch
torch.cuda.empty_cache()
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import LeakyReLU
from torch.nn import LogSoftmax
from torch.nn import Softmax
from torch import flatten
from torch.nn import NLLLoss
from torch.nn import CrossEntropyLoss

class SimpleConvModel(Module):
    def __init__(self, numChannels, classes):
        super(SimpleConvModel, self).__init__()

        self.conv_1 = Conv2d(in_channels=numChannels, out_channels=20, kernel_size=(5, 5), stride=3, dtype=torch.float16)
        self.relu_1 = LeakyReLU()
        self.max_pool_1 = MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.conv_2 = Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5), stride=3, dtype=torch.float16)
        self.relu_2 = LeakyReLU()
        self.max_pool_2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc_1 = Linear(in_features=35700, out_features=500, dtype=torch.float16)
        self.relu_3 = LeakyReLU()

        self.fc_2 = Linear(in_features=500, out_features=classes, dtype=torch.float16)
        self.softmax = Softmax(dim=1)

        self._loss = CrossEntropyLoss()

    def forward(self, x):
        # convert to float16
        x = x.to(torch.float16)
        # print(x.shape)
        x = self.conv_1(x)
        # print(x.shape)
        x = self.relu_1(x)
        x = self.max_pool_1(x)
        # print(x.shape)

        x = self.conv_2(x)
        # print(x.shape)
        x = self.relu_2(x)
        x = self.max_pool_2(x)
        # print(x.shape)

        x = flatten(x, 1)
        # print(x.shape)

        x = self.fc_1(x)
        x = self.relu_3(x)
        # print(x.shape)

        x = self.fc_2(x)
        # print(x.shape)

        output = self.softmax(x)
        # print(output.shape)
        return output

    def loss(self, logits, labels):
        return self._loss(logits, labels.to(torch.float16))

    def accuracy(self, logits, labels):
        correct = (logits == labels).float().sum()
        return 100 * correct / len(logits)

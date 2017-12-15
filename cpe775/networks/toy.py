from torch import nn
from torch.nn import init
import torch.nn.functional as F


class BaseNet(nn.Module):

    def __init__(self, input_shape, output_shape):
        super(BaseNet, self).__init__()

        self.output_shape = output_shape
        self.input_shape = input_shape

        C, W, H = input_shape
        P, D = output_shape

        self.features = nn.Sequential(
            nn.Linear(C*W*H, 300),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(300, P*D)
        )

    def forward(self, x):
        C, W, H = self.input_shape
        P, D = self.output_shape

        x = x.view(-1, C*W*H) # C x W x H -> C * W * H

        x = self.features(x)

        x = self.classifier(x)


        return x.view(-1, P, D)


class CNNNet(nn.Module):

    def __init__(self, input_shape, output_shape):
        super(CNNNet, self).__init__()

        self.output_shape = output_shape
        self.input_shape = input_shape

        C, W, H = input_shape
        P, D = output_shape

        self.features = nn.Sequential(
            nn.Conv2d(C, 32, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(128, 256, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
        )

        self.classifier = nn.Sequential(
            nn.Linear(int(W/16*H/16*256), 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, P*D)
        )

    def forward(self, x):
        C, W, H = self.input_shape
        P, D = self.output_shape

        x = self.features(x)

        x = x.view(-1, int(W/16*H/16*256))

        x = self.classifier(x)

        return x.view(-1, P, D)

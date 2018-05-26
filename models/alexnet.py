"""AlexNet architecture pytorch model."""

from torch import nn

from models import blocks


class AlexNet(nn.Module):
    """This is the original AlexNet architecture, and not the version introduced
    in the "one weird trick" paper."""
    LR_REGIME = [1, 140, 0.01, 141, 170, 0.001, 171, 200, 0.0001]

    def __init__(self):
        super().__init__()
        self.conv1 = blocks.Conv2dBnRelu(3, 96, 11, 4, 2,
                                         pooling=nn.MaxPool2d(2))
        self.conv2 = blocks.Conv2dBnRelu(96, 256, 5, 1, 2,
                                         pooling=nn.MaxPool2d(2))
        self.conv3 = blocks.Conv2dBnRelu(256, 384, 3, 1, 1)
        self.conv4 = blocks.Conv2dBnRelu(384, 384, 3, 1, 1)
        self.conv5 = blocks.Conv2dBnRelu(384, 256, 3, 1, 1,
                                         pooling=nn.MaxPool2d(2))

        self.fc6 = blocks.LinearBnRelu(256 * 6 * 6, 4096)
        self.fc7 = blocks.LinearBnRelu(4096, 4096)
        self.fc8 = nn.Linear(4096, 1000, bias=False)

    def convolutions(self, x):
        return nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4,
                             self.conv5)(x)

    def fully_connecteds(self, x):
        return nn.Sequential(self.fc6, self.fc7, self.fc8)(x)

    def forward(self, x):
        x = self.convolutions(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connecteds(x)
        return x

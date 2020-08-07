import torch
from torch import nn
from torch.nn import functional as F

class LeNet5(nn.Module):
    """
    for cifar10 dataset.
    """

    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv_unit = nn.Sequential(
            # x: [b,3,32,32] -> [b,6,28,28]
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            #
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

        )

        # flatten
        # fc unit
        self.fc_unit = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

        # [b,3,32,32]
        temp = torch.randn(2, 3, 32, 32)
        out = self.conv_unit(temp)
        # [b,16,5,5]
        print('conv out:', out.shape)

        # use Cross Entropy Loss ( or MSE -> nn.MSELoss() )
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, x):
        """
        :param x: [b,3,32,32]
        :return:
        """
        batch_size = x.size(0) # size(0)返回[b,3,32,32]的b
        # [b,3,32,32] -> [b,16,5,5]
        x = self.conv_unit(x)
        # [b,16,5,5] -> [b,16*6*6] Flatten
        x = x.view(batch_size, 16*5*5)
        # [b,16*5*5] -> [b,10]
        logits = self.fc_unit(x)

        return logits


def main():
    net = LeNet5()
    temp = torch.randn(2,3,32,32)
    out = net(temp)
    print('LeNet5 out: ',out.shape)


if __name__ == '__main__':
    main()

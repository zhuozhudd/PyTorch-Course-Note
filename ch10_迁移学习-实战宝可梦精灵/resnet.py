import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    """
    resnet block
    """

    def __init__(self, channel_in, channel_out, stride=1):
        """

        :param channel_in:
        :param channel_out:
        """
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(channel_out)
        self.conv2 = nn.Conv2d(channel_out, channel_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_out)

        self.extra = nn.Sequential()
        if channel_out != channel_in:
            # [b,ch_in,h,w] -> [b,ch_out,h,w]
            self.extra = nn.Sequential(
                nn.Conv2d(channel_in, channel_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(channel_out)
            )

    def forward(self, x):
        """
        :param x: [b, ch, h, w]
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # short cut:
        # extra module : [b,ch_in,h,w] with [b,ch_out,h,w]
        # element - wise add
        out = self.extra(x) + out
        out = F.relu(out)
        return out


class ResNet18(nn.Module):

    def __init__(self, num_class):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(16)
        )
        # followed 4 blocks
        self.blk1 = ResBlock(16, 32, stride=3)
        self.blk2 = ResBlock(32, 64, stride=3)
        self.blk3 = ResBlock(64, 128, stride=2)
        self.blk4 = ResBlock(128, 256, stride=2)

        self.outlayer = nn.Linear(256 * 3 * 3, num_class)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))
        # [b,64,h,w] -> [b,1024,h,w]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        x = x.view(x.size(0), -1)
        x = self.outlayer(x)

        return x


def main():
    blk = ResBlock(64, 128)
    temp = torch.randn(2, 64, 224, 224)
    out = blk(temp)
    print('block: ', out.shape)

    model = ResNet18(5)
    temp = torch.randn(2, 3, 224, 224)
    out = model(temp)
    print('resnet:', out.shape)

    p = sum(map(lambda p: p.numel(), model.parameters()))
    print('parameters size: ', p)


if __name__ == '__main__':
    main()

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import torchvision
import matplotlib.pyplot as plt
from utils import plot_image, plot_curve, ont_hot

# step1. load dataset

batch_size = 64
train_load = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
    'mnist data',
    train=True,
    download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307, ), (0.3081, ))
    ])),
                                         batch_size=batch_size,
                                         shuffle=True)

test_load = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
    'mnist data/',
    train=False,
    download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307, ), (0.3081, ))
    ])),
                                        batch_size=batch_size,
                                        shuffle=False)

x, y = next(iter(train_load))
print(x.shape, y.shape)
print(x.min(), x.max())
plot_image(x, y, 'image_sample')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # xw+b
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # x: [b,1,28,28]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.softmax(self.fc3(x))
        x = self.fc3(x)  # h3 = h2w3 + b3
        return x


net = Net()
# net.parameters: [w1,b1,w2,b2,w3,b3]
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

train_loss = []

for epoch in range(3):
    for batch_idx, (x, y) in enumerate(train_load):
        # x:[b,1,28,28] , y:[64]
        # net只接收[b,feature] 需要 [b,1,28,28] -> [b,feature]
        x = x.view(x.size(0), -1)
        out = net(x)  # -> [b,10]
        # [b,10]
        y_onehot = ont_hot(y)
        # loss = mse(out,y_onehot)
        loss = F.mse_loss(out, y_onehot)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # w' = w - lr * grad
        train_loss.append(loss.item())
        if batch_idx % 10 == 0:
            print(epoch, batch_idx, loss.item())

plot_curve(train_loss)
# we get optimal [w1,b1,w2,b2,w3,b3]

total_correct = 0
for x, y in test_load:
    x = x.view(x.size(0), -1)
    out = net(x)
    # out:[b,10] -> predict_value:[b]
    predict_value = out.argmax(dim=1)
    correct = predict_value.eq(
        y).sum().float().item()  # 正确个数,转为float类型,转为python数值类型
    total_correct += correct

total_num = len(test_load.dataset)
acc = total_correct / total_num
print('test acc:', acc)

x, y = next(iter(test_load))
out = net(x.view(x.size(0), -1))
pred = out.argmax(dim=1)
plot_image(x, pred, 'test')

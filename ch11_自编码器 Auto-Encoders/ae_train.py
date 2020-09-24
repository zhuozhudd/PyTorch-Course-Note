import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch import nn, optim

from ae import AE
from vae import VAE
import visdom


def main():
    mnist_train = datasets.MNIST('./../data', train=True, transform=transforms.Compose([
        transforms.ToTensor()
    ]), download=False)
    mnist_train = DataLoader(mnist_train, batch_size=128, shuffle=True)

    mnist_test = datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor()
    ]), download=False)
    mnist_test = DataLoader(mnist_test, batch_size=128, shuffle=False)

    # 无监督学习，不需要label
    x, _ = iter(mnist_train).next()
    print('x: ', x.shape)

    device = torch.device('cuda')

    # model = AE().to(device)
    model = VAE().to(device)

    print(model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    viz = visdom.Visdom()

    for epoch in range(100):

        for batchidx, (x, _) in enumerate(mnist_train):
            # [b,1,28,28]
            x = x.to(device)

            x_hat = model(x)
            loss = criterion(x_hat, x)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch: ', epoch, '  loss:', loss.item())

        x, _ = iter(mnist_test).next()
        x = x.to(device)
        with torch.no_grad():
            x_hat = model(x)
        viz.images(x, nrow=8, win='x', opts=dict(title='x'))
        viz.images(x_hat, nrow=8, win='x_hat', opts=dict(title='x_hat'))


if __name__ == '__main__':
    main()

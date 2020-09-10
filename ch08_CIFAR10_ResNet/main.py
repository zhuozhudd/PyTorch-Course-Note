import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from ResNet import ResNet18


def main():
    batch_size = 32

    cifar_train = datasets.CIFAR10('../data/', train=True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]), download=True)

    cifar_train = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)

    cifar_test = datasets.CIFAR10('../data/', train=False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)

    cifar_test = DataLoader(cifar_test, batch_size=batch_size, shuffle=True)

    x, label = iter(cifar_train).next()
    print('x: ', x.shape, 'label: ', label.shape)

    device = torch.device('cuda')
    model = LeNet5().to(device)
    # model = LeNet5()
    # model = ResNet18()
    print(model)
    # criterion = nn.CrossEntropyLoss().to(device)
    criterion = nn.CrossEntropyLoss()  # 包含了softmax
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(10):

        for batch_idx, (x, label) in enumerate(cifar_train):
            # x,label = x.to(device), label.to(device)
            # x:[b,3,32,32] label:[b]
            logits = model(x)
            # pred 是 logits 经过 softmax 得到的
            # label: [b]  logits: [b,10]  loss: tensor scalar
            loss = criterion(logits, label)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 完成一个epoch
        print('epoch: ', epoch, 'loss: ', loss.item())

        model.eval()  # -> test模式
        with torch.no_grad():
            # test
            # test 无需构建计算图，no_grad()
            total_correct = 0
            total_num = 0
            for x, label in cifar_test:
                # [b,10]
                logits = model(x)
                pred = logits.argmax(dim=1)
                total_correct += torch.eq(pred, label).float().sum().item()
                total_num += x.size(0)

            acc = total_correct / total_num
            print('epoch：', epoch, 'acc: ', acc)


if __name__ == '__main__':
    main()

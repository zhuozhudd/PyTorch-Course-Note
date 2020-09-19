import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from ResNet import ResNet18
# from LeNet5 import LeNet5


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
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]), download=True)

    cifar_test = DataLoader(cifar_test, batch_size=batch_size, shuffle=True)

    x, label = iter(cifar_train).__next__()
    print('x: ', x.shape, 'label: ', label.shape)

    device = torch.device('cuda')

    # model = LeNet5().to(device)
    model = ResNet18().to(device)

    criterion = nn.CrossEntropyLoss().to(device)  # 包含了softmax
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(model)

    for epoch in range(100):
        model.train()
        for batch_size, (x, label) in enumerate(cifar_train):
            # x:[b,3,32,32] label:[b]
            x, label = x.to(device), label.to(device)
            logits = model(x)
            # logits:[b,10]  label:[b]
            loss = criterion(logits, label)  # loss:tensor scalar长度为0的标量

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch: ', epoch, '  loss:', loss.item())  # 将loss转换为numpy类型打印出来

        # test
        model.eval()
        with torch.no_grad():

            total_correct = 0
            total_num = 0
            for x, label in cifar_test:
                x, label = x.to(device), label.to(device)

                # logits: [b,10]
                logits = model(x)
                # pred: [b]
                pred = logits.argmax(dim=1)
                # [b] vs [b]  -> scalar tensor
                total_correct += torch.eq(pred, label).float().sum().item()
                total_num += x.size(0)

            acc = total_correct / total_num
            print('acc: ', acc)


if __name__ == '__main__':
    main()

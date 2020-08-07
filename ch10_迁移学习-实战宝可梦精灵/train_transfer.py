import torch
import visdom
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from pokemon import Pokemon
from utils import Flatten

# torchvision中的resnet18自带已经训练好的权重
# from resnet import ResNet18


batch_size = 32
lr = 1e-3
epoches = 10

# device = torch.device('cuda')
torch.manual_seed(1234)

train_db = Pokemon('../data/pokemon', 224, mode='train')
val_db = Pokemon('../data/pokemon', 224, mode='val')
test_db = Pokemon('../data/pokemon', 224, mode='test')
train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_db, batch_size=batch_size, num_workers=2)
test_loader = DataLoader(test_db, batch_size=batch_size, num_workers=2)

viz = visdom.Visdom()


def evalute(model, loader):
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        # x,y = x.to(device),y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()
    return correct / total


def main():
    trained_model = resnet18(pretrained=True)
    # 用children()方法取resnet18前17层的权重，转换成list，[:-1]即前17层，
    # 由于Sequential接收的是打散的数据，所以加*
    model = nn.Sequential(*list(trained_model.children())[:-1],
                          Flatten(),  # [b,512,1,1]->[b,512]
                          nn.Linear(512, 5)
                          )  # 迁移学习
    # model = model.to(device)
    # x = torch.randn(2,3,224,224)
    # print(model(x).shape)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))

    for epoch in range(epoches):
        for step, (x, y) in enumerate(train_loader):
            # x:[b,3,224,224,] y:[b]
            # x,y = x.to(device),y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            viz.line([loss.item()], [global_step], win='loss', update='append')
            global_step += 1

        # validation
        if epoch % 2 == 0:
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best.mdl')  # 保存模型

                viz.line([val_acc], [global_step], win='loss', update='append')

    print('best acc: ', best_acc, ' best epoch: ', best_epoch)
    model.load_state_dict(torch.load('best.mdl'))
    print('loader from ckpt!')

    test_acc = evalute(model, test_loader)
    print('test acc:', test_acc)


if __name__ == '__main__':
    main()

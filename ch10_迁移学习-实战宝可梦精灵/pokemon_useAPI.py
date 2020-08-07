import time

import torchvision
import visdom
from torch.utils.data import DataLoader
from torchvision import transforms


def main():
    viz = visdom.Visdom()

    tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    db = torchvision.datasets.ImageFolder(root='../data/pokemon', transform=tf)

    # num_workers=8 8线程加速，让batch_size过大时可以使用
    loader = DataLoader(db, batch_size=32, shuffle=True)
    # num_workers=8 8线程加速，让batch_size过大时可以使用
    # loader = DataLoader(db, batch_size=32, shuffle=True, num_workers=8)

    print(db.class_to_idx)  # name2label信息

    # 批处理后，每次读取32张图
    for x, y in loader:
        viz.images(x, nrow=8, win='batch', opts=dict(title='batch'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))
        time.sleep(10)
        print('one batch')


if __name__ == '__main__':
    main()

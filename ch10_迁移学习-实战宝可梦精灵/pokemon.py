import csv
import glob
import os
import random
import time

import torch
import torchvision
import visdom
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class Pokemon(Dataset):
    def __init__(self, root, resize, mode):
        super(Pokemon, self).__init__()

        self.root = root
        self.resize = resize

        self.name2label = {}
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            self.name2label[name] = len(self.name2label.keys())

        # print(self.name2label)
        # image <-> label
        self.images, self.labels = self.load_csv('images.csv')

        # 划分数据
        if mode == 'train':  # train 60% : 0~60%
            self.images = self.images[:int(0.6 * len(self.images))]
            self.labels = self.labels[:int(0.6 * len(self.labels))]

        elif mode == 'val':  # val 20% : 60%~80%
            self.images = self.images[int(0.6 * len(self.images)):int(0.8 * len(self.images))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]

        else:  # test 20% : 80%~100%
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]

    def load_csv(self, filename):

        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                # 'pokemon\\mewtwo\\00001.png'
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
            # 1165 '../data/pokemon/squirtle/00000170.png'
            print(len(images), images)

            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:  # '../data/pokemon/squirtle/00000170.png'
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    # '../data/pokemon/squirtle/00000170.png' , 4
                    writer.writerow([img, label])

                print('writen into csv file:', filename)

        # read from csv file
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)
                images.append(img)
                labels.append(label)

        assert len(images) == len(labels)
        return images, labels

    def __len__(self):
        return len(self.images)

    def denormalize(self, x_hat):
        # 由于visdom接收的图片是0～1，正则化后是-1~1，显示效果不好，所以显示时需要denormalize
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # x_hat = (x-mean)/std
        # x:[c,h,w]  mean:[3] -> [3,1,1] -> broadcast
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x = x_hat * std + mean
        return x

    def __getitem__(self, idx):
        # idx : [0~len(images)]
        # 这里的img只是一个路径
        img, label = self.images[idx], self.labels[idx]
        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),  # 把路径转换为RGB图
            transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
            transforms.RandomRotation(15),  # 旋转15度
            transforms.CenterCrop(self.resize),  # 中心裁剪
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img = tf(img)
        label = torch.tensor(label)  # 将label转换为tensor
        return img, label


def main():
    viz = visdom.Visdom()

    db = Pokemon('../data/pokemon', 224, 'train')
    x, y = next(iter(db))
    print('sample: ', x.shape, y.shape, y)
    viz.image(db.denormalize(x), win='sample_x', opts=dict(title='sample_x'))

    loader = DataLoader(db, batch_size=32, shuffle=True)
    # 批处理后，每次读取32张图
    for x, y in loader:
        viz.images(db.denormalize(x), nrow=8, win='batch', opts=dict(title='batch'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))
        time.sleep(10)


if __name__ == '__main__':
    main()

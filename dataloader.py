from torchvision import transforms, utils
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image
import os, sys


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    def __init__(self, csv, dataset, transform=None, loader=default_loader):
        fh = open(csv, 'r')
        imgs = []
        for line in fh.readlines()[1:]:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split(',')
            imgs.append((words[0], words[1]))
        self.dataset = dataset
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        ds = os.path.join('./dataset', self.dataset)
        fn = os.path.join(ds, fn)
        img = self.loader(fn)
        if self.transform is not None:
            img, label = self.transform(img, label)
        return img, label

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    train_dataset = MyDataset('train.csv')


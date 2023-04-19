import torch
from torch.utils.data import DataLoader

from preprocess import ImageDataset


def main():
    x = torch.rand(5, 3)
    print(x)

    train_dataset = ImageDataset(directory_path='/var/project/train_data')
    test_dataset = ImageDataset(directory_path='/var/project/test_data')



    train_dataloader = DataLoader(dataset=train_dataset, batch_size=256)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=256)

    for i in range(10):
        image_tensor, label = next(iter(train_dataloader))
        print(image_tensor, label)

    for i in range(10):
        image_tensor, label = next(iter(test_dataloader))
        print(image_tensor, label)

if __name__ == '__main__':
    main()

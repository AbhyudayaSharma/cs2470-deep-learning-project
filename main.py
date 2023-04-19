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

    for image_tensor, label in train_dataloader[:10] :
      print(image_tensor, label)

    for image_tensor, label in test_dataloader[:10] :
      print(image_tensor, label)

if __name__ == '__main__':
    main()

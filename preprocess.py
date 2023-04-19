import os
import random

import torch
import torchvision

from torch.utils.data import DataLoader
from torchvision.io.image import ImageReadMode

class ImageDataset(torch.utils.data.IterableDataset):

    def __init__(self, directory_path) -> None:
        super(ImageDataset).__init__()

        self.directory_path = directory_path
        self.image_paths = list(map(lambda x: x.name, os.scandir(path=directory_path)))

    def __iter__(self) -> None:
        # shuffle the image paths
        random.shuffle(self.image_paths)
        # get name of next image
        for path in self.image_paths:
            # extract label from image name
            label = path.split(".")[0]
            label = label[: label.rindex('_')]
            # return image as tensor and label
            yield torchvision.io.read_image(os.path.join(self.directory_path, path), mode=ImageReadMode.RGB), label

import os
import random

import torch
import torchvision

from torch.utils.data import DataLoader
from torchvision.io.image import ImageReadMode

class ImageDataset(torch.utils.data.IterableDataset):

    def __init__(self, directory_path) -> None:
        super(ImageDataset).__init__()

        self.image_paths = os.scandir(path=directory_path)

    def __iter__(self) -> None:
        # get name of next image
        for path in random.shuffle(self.image_paths):
            # extract label from image name
            label = path.split(".")[0]
            label = label[: label.rindex('_')]
            # return image as tensor and label
            yield torchvision.io.read_image(path, mode=ImageReadMode.RGB), label

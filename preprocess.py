import os
import random

import torch
torch.cuda.empty_cache()
import torchvision

from torch.utils.data import DataLoader
from torchvision.io.image import ImageReadMode


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, directory_path) -> None:
        super(ImageDataset).__init__()

        self.directory_path = directory_path
        self.image_paths = list(map(lambda x: x.name, os.scandir(path=directory_path)))

        countries = []
        for path in self.image_paths:
            # extract label from image name
            label = path.split(".")[0]
            label = label[: label.rindex("_")]
            countries.append(label)

        self.country_labels = list(sorted(set(countries)))
        self.label_map = {val: i for i, val in enumerate(self.country_labels)}

    def __iter__(self) -> None:
        # shuffle the image paths
        random.shuffle(self.image_paths)
        # get name of next image
        for path in self.image_paths:
            # extract label from image name
            label = path.split(".")[0]
            label = label[: label.rindex("_")]
            # return image as tensor and label
            try :
                yield torchvision.io.read_image(
                    os.path.join(self.directory_path, path), mode=ImageReadMode.RGB
                ), label
            except RuntimeError as e:
                print(path)
                raise e

import sys
from pprint import pprint

import torch
import torchvision
from torchvision.io import ImageReadMode

from preprocess import ImageDataset


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_dataset = ImageDataset(directory_path='/var/project/test_data')

    model = torch.load(sys.argv[1])
    img = torchvision.io.read_image(sys.argv[2], mode=ImageReadMode.RGB)

    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()

        # get X and Y
        x = torch.unsqueeze(img, 0) / 255.0
        x = x.to(device)

        # get model predictions for this batch
        logits = model(x)
        predictions = torch.nn.functional.softmax(logits, dim=1)
        probabilities, topk_catid = torch.topk(predictions, 5)
        probabilities, topk_catid = probabilities[0], topk_catid[0]
        pprint(list(zip(probabilities, map(lambda id: test_dataset.country_labels[id], topk_catid))))


if __name__ == '__main__':
    main()

import datetime

import torch
import torchvision

torch.cuda.empty_cache()
from datetime import datetime
from torch.utils.data import DataLoader
import gc
import os

from preprocess import ImageDataset

from torch.optim import Adam
from torch.nn.functional import one_hot


def correct_predictions(truth, predictions, top_k=3):
    count = 0

    _, top5_catid = torch.topk(predictions, top_k)
    for i in range(truth.shape[0]):
        if truth[i] in top5_catid[i]:
            count += 1

    return count


def save_model(model):
    save_path = '/var/project/models'
    # model_uuid = uuid.uuid4()
    file_path = os.path.join(save_path, f'model_{datetime.now().isoformat()}_.pickle')
    with open(file_path, 'wb') as f:
        torch.save(model, file_path)


def load_model(path):
    return torch.load(path)


def main():
    device = torch.device('cuda')

    BATCH_SIZE = 7
    EPOCHS = 4
    train_dataset = ImageDataset(directory_path='/var/project/train_data')
    test_dataset = ImageDataset(directory_path='/var/project/test_data')

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

    train_data_count = 25249
    test_image_count = 6283

    model = torchvision.models.resnet50(num_classes=43)
    model = model.to(device)
    print(model)

    # initialize our optimizer and loss function
    opt = Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    # initialize a dictionary to store training history
    H = {"train_loss": [], "train_acc": []}

    # loop over our epochs
    for e in range(0, EPOCHS):
        model.train()
        total_train_loss = 0

        num_correct_train_predictions = 0

        train_steps = 0
        for x, y in iter(train_dataloader):
            classes = torch.Tensor(list(map(lambda label: train_dataset.label_map[label], y))).to(torch.int64)
            y = one_hot(classes, num_classes=43)
            x = x / 255.0
            y = y.to(torch.float16)
            x, y = x.to(device), y.to(device)
            # zero out the gradients, perform the backpropagation step,and update the weights
            opt.zero_grad()

            pred = model(x)
            print(f'{e} {train_steps}')
            loss = loss_fn(pred, y)

            loss.backward()
            opt.step()

            total_train_loss += loss

            # accuracy(y_original, pred)
            # num_correct_train_predictions += (
            #     (torch.nn.functional.softmax(pred[0], dim=0).argmax(1) == y.argmax(1)).type(torch.float16).sum().item()
            # )
            num_correct_train_predictions += correct_predictions(classes, torch.nn.functional.softmax(pred, dim=1))

            train_steps += 1
            gc.collect()
            break

        # calculate the average training loss and accuracy
        avg_train_loss = total_train_loss / train_steps
        avg_train_acc = num_correct_train_predictions / train_data_count

        # update our training history
        H["train_loss"].append(avg_train_loss)
        H["train_acc"].append(avg_train_acc)
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
        print(
            "Train loss: {:.6f}, Train accuracy: {:.4f}".format(
                avg_train_loss, avg_train_acc
            )
        )

    save_model(model)

    total_test_loss = 0
    num_correct_test_predictions_top1 = 0
    num_correct_test_predictions_top2 = 0
    num_correct_test_predictions_top3 = 0
    num_correct_test_predictions_top5 = 0

    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()
        for x, y in iter(test_dataloader):
            classes = torch.Tensor(list(map(lambda label: train_dataset.label_map[label], y))).to(torch.int64)
            y = one_hot(classes, num_classes=43)

            x = x / 255.0
            y = y.to(torch.float16)

            # send the input to the device
            x, y = x.to(device), y.to(device)

            pred = model(x)
            total_test_loss += loss_fn(pred, y)
            num_correct_test_predictions_top1 += correct_predictions(classes, torch.nn.functional.softmax(pred, dim=1), 1)
            num_correct_test_predictions_top2 += correct_predictions(classes, torch.nn.functional.softmax(pred, dim=1), 2)
            num_correct_test_predictions_top3 += correct_predictions(classes, torch.nn.functional.softmax(pred, dim=1), 3)
            num_correct_test_predictions_top5 += correct_predictions(classes, torch.nn.functional.softmax(pred, dim=1), 5)
            gc.collect()
            break

        test_accuracy1 = num_correct_test_predictions_top1 / test_image_count
        test_accuracy2 = num_correct_test_predictions_top2 / test_image_count
        test_accuracy3 = num_correct_test_predictions_top3 / test_image_count
        test_accuracy5 = num_correct_test_predictions_top5 / test_image_count

        print(
            "Test loss: {:.6f}, Test accuracy 1: {:.4f}, Test accuracy 2: {:.4f}, Test accuracy 3: {:.4f}, Test accuracy 5: {:.4f}".format(
                total_test_loss, test_accuracy1, test_accuracy2, test_accuracy3, test_accuracy5
            )
        )


if __name__ == "__main__":
    main()

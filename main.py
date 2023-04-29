import torch
import torchvision
torch.cuda.empty_cache()
from torch.utils.data import DataLoader
import gc

from preprocess import ImageDataset
from model import SimpleConvModel

from torch.optim import Adam
from torch.nn.functional import one_hot


def correct_predictions(truth, predictions, top_k=3):
    count = 0
    _, top5_catid = torch.topk(predictions, top_k)
    for i in range(predictions.shape[0]):
        if truth[i] in top5_catid[i]:
            count += 1

    return count

def main():
    device = torch.device('cuda')
    x = torch.rand(5, 3)
    print(x)

    BATCH_SIZE = 5
    EPOCHS = 4
    train_dataset = ImageDataset(directory_path='/var/project/train_data')
    test_dataset = ImageDataset(directory_path='/var/project/test_data')

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

    train_data_count = 25249
    test_image_count = 6283

    # for i in range(10):
    #     image_tensor, label = next(iter(train_dataloader))
    #     print(image_tensor, label)
    #
    # for i in range(10):
    #     image_tensor, label = next(iter(test_dataloader))
    #     print(image_tensor, label)

    # model = SimpleConvModel(numChannels=3, classes=43)
    model = torchvision.models.resnet50(num_classes=43)
    model = model.to(device)

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
            y_original = y
            y = one_hot(
                torch.Tensor(list(map(lambda label: train_dataset.label_map[label], y))).to(torch.int64),
                num_classes=43,
            )
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
            num_correct_train_predictions += correct_predictions(y_original, torch.nn.functional.softmax(pred))

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

    total_test_loss = 0
    num_correct_test_predictions = 0

    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()
        for x, y in iter(test_dataloader):
            y_original = y
            y = one_hot(
                torch.Tensor(list(map(lambda label: test_dataset.label_map[label], y))).to(torch.int64),
                num_classes=43,
            )

            x = x / 255.0
            y = y.to(torch.float16)

            # send the input to the device
            x, y = x.to(device), y.to(device)

            pred = model(x)
            total_test_loss += loss_fn(pred, y)
            num_correct_test_predictions += correct_predictions(y_original, torch.nn.functional.softmax(pred))
            # num_correct_test_predictions += (
            #     (torch.nn.functional.softmax(pred[0], dim=0).argmax(1) == y.argmax(1)).type(torch.float16).sum().item()
            # )

            gc.collect()
            break

        test_accuracy = num_correct_test_predictions / test_image_count
        print(
            "Test loss: {:.6f}, Test accuracy: {:.4f}".format(
                total_test_loss, test_accuracy
            )
        )


if __name__ == "__main__":
    main()

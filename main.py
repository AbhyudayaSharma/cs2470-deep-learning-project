import torch
from torch.utils.data import DataLoader

from preprocess import ImageDataset
from model import SimpleConvModel

from torch.optim import Adam
from torch.nn.functional import one_hot


def main():
    device = torch.device('cuda')
    x = torch.rand(5, 3)
    print(x)

    BATCH_SIZE = 64
    EPOCHS = 1
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

    model = SimpleConvModel(numChannels=3, classes=43)
    model = model.to(device)
    # initialize our optimizer and loss function
    opt = Adam(model.parameters(), lr=1e-3)
    loss_fn = model.loss

    # initialize a dictionary to store training history
    H = {"train_loss": [], "train_acc": []}

    # loop over our epochs
    for e in range(0, EPOCHS):
        model.train()

        total_train_loss = 0
        total_test_loss = 0

        num_correct_train_predictions = 0
        num_correct_test_predictions = 0

        train_steps = 0
        for x, y in iter(train_dataloader):
            y = one_hot(
                torch.Tensor(list(map(lambda label: train_dataset.label_map[label], y))).to(torch.int64),
                num_classes=43,
            )
            x, y = x.to(device), y.to(device)

            # zero out the gradients, perform the backpropagation step,and update the weights
            opt.zero_grad()

            pred = model(x)
            loss = loss_fn(pred, y)

            loss.backward()
            opt.step()

            total_train_loss += loss
            num_correct_train_predictions += (
                (pred.argmax(1) == y).type(torch.float).sum().item()
            )

            train_steps += 1

        batch_size_entries = y.shape[0]
        # calculate the average training loss and accuracy
        avg_train_loss = total_train_loss / train_steps
        avg_train_acc = num_correct_train_predictions / batch_size_entries

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

    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()
        for x, y in iter(test_dataloader):
            y = one_hot(
                torch.Tensor(list(map(lambda label: test_dataset.label_map[label], y))).to(torch.int64),
                num_classes=43,
            )

            # send the input to the device
            x, y = x.to(device), y.to(device)

            pred = model(x)
            total_test_loss += loss_fn(pred, y)
            num_correct_test_predictions += (
                (pred.argmax(1) == y).type(torch.float).sum().item()
            )
        test_accuracy = num_correct_test_predictions / test_image_count
        print(
            "Test loss: {:.6f}, Test accuracy: {:.4f}".format(
                total_test_loss, test_accuracy
            )
        )


if __name__ == "__main__":
    main()

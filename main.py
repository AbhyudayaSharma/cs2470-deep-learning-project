import os
import gc
import torch
import torchvision
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.functional import one_hot
from preprocess import ImageDataset

torch.cuda.empty_cache()

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
    file_path = os.path.join(save_path, f'model_{datetime.now().isoformat()}.pickle')
    with open(file_path, 'wb') as f:
        torch.save(model, file_path)


def load_model(path):
    return torch.load(path)


def main():
    device = torch.device('cuda')

    # set command line argument
    BATCH_SIZE = 5
    EPOCHS = 10
    LEARNING_RATE = 1e-3

    # get datasets
    train_dataset = ImageDataset(directory_path='/var/project/train_data')
    test_dataset = ImageDataset(directory_path='/var/project/test_data')

    # load data
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

    # count number of images
    train_data_count = 25248
    test_image_count = 6283

    # define model
    # model = torchvision.models.resnet50(num_classes=43)   # drop weights
    # model = torchvision.models.DenseNet(num_classes=43)
    model = torchvision.models.Inception3(num_classes=43)
    model = model.to(device)
    # show model architecture
    print(model)

    # initialize optimizer and loss function
    opt = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()

    # loop over the training epochs
    for e in range(0, EPOCHS):

        # set model in train mode
        model.train()

        # track total loss and correct predictions
        total_train_loss = 0
        num_correct_train_predictions = 0

        train_steps = 0
        for x, y in iter(train_dataloader):
            print(f'{e} {train_steps}')

            # get X and Y
            classes = torch.Tensor(list(map(lambda label: train_dataset.label_map[label], y))).to(torch.int64)
            y = one_hot(classes, num_classes=43)
            x = x / 255.0
            y = y.to(torch.float16)
            x, y = x.to(device), y.to(device)

            # zero out the gradients
            opt.zero_grad()

            # get model predictions for this batch
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            # calculate loss for this batch
            loss = loss_fn(logits, y)

            # perform the backpropagation step
            loss.backward()
            # update the weights
            opt.step()

            # update the total loss and number of predictions
            total_train_loss += loss
            num_correct_train_predictions += correct_predictions(classes, torch.nn.functional.softmax(logits, dim=1), top_k=1)

            # increment counter and run garbage collector
            train_steps += 1
            gc.collect()
            # break

        # calculate the average training loss and accuracy
        avg_train_loss = total_train_loss / train_steps
        avg_train_acc = num_correct_train_predictions / train_data_count

        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
        print(
            "Train loss: {:.6f}, Train accuracy: {:.4f}".format(
                avg_train_loss, avg_train_acc
            )
        )

    # save the model for future use
    save_model(model)

    total_test_loss = 0
    num_correct_test_predictions_top1 = 0
    num_correct_test_predictions_top2 = 0
    num_correct_test_predictions_top3 = 0
    num_correct_test_predictions_top5 = 0

    test_steps = 0
    with torch.no_grad():

        # set the model in evaluation mode
        model.eval()

        # test it on the test dataset
        for x, y in iter(test_dataloader):
            print(f' {test_steps}')

            # get X and Y
            classes = torch.Tensor(list(map(lambda label: test_dataset.label_map[label], y))).to(torch.int64)
            y = one_hot(classes, num_classes=43)
            x = x / 255.0
            y = y.to(torch.float16)
            # send the input to the device
            x, y = x.to(device), y.to(device)

            # get model predictions for this batch
            logits = model(x)
            # calculate loss for this batch
            loss = loss_fn(logits, y)

            # update the total loss and number of predictions
            total_test_loss += loss
            num_correct_test_predictions_top1 += correct_predictions(classes, torch.nn.functional.softmax(logits, dim=1), top_k=1)
            num_correct_test_predictions_top2 += correct_predictions(classes, torch.nn.functional.softmax(logits, dim=1), top_k=2)
            num_correct_test_predictions_top3 += correct_predictions(classes, torch.nn.functional.softmax(logits, dim=1), top_k=3)
            num_correct_test_predictions_top5 += correct_predictions(classes, torch.nn.functional.softmax(logits, dim=1), top_k=5)

            # increment counter and run garbage collector
            test_steps += 1
            gc.collect()
            # break

        # calculate the average training loss and accuracy
        avg_test_loss = total_test_loss / test_steps
        test_accuracy1 = num_correct_test_predictions_top1 / test_image_count
        test_accuracy2 = num_correct_test_predictions_top2 / test_image_count
        test_accuracy3 = num_correct_test_predictions_top3 / test_image_count
        test_accuracy5 = num_correct_test_predictions_top5 / test_image_count

        print(
            "Test loss: {:.6f}, Test accuracy 1: {:.4f}, Test accuracy 2: {:.4f}, Test accuracy 3: {:.4f}, Test accuracy 5: {:.4f}".format(
                avg_test_loss, test_accuracy1, test_accuracy2, test_accuracy3, test_accuracy5
            )
        )


if __name__ == "__main__":
    main()

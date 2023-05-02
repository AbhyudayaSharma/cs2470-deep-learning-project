import gc
import torch
import clip
from torch.utils.data import DataLoader
from preprocess import ImageDataset
from sklearn.linear_model import LogisticRegression

torch.cuda.empty_cache()

def correct_predictions(logits, predictions, top_k=3):
    count = 0

    _, top5_catid = torch.topk(predictions, top_k)
    for i in range(len(logits)):
        if logits[i] in top5_catid[i]:
            count += 1

    return count

def clip_module():

    # set command line argument
    BATCH_SIZE = 512

    # define model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model = model.to(device)
    # show model architecture
    print(model)

    # get datasets
    train_dataset = ImageDataset(directory_path='/var/project/train_data', clip_preprocessing = preprocess)
    test_dataset = ImageDataset(directory_path='/var/project/test_data', clip_preprocessing = preprocess)

    # load data
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

    # count number of images
    train_data_count = 25248
    test_image_count = 6283

    # get training features
    train_features = []
    train_labels = []
    with torch.no_grad():

        train_steps = 0
        for x, y in iter(train_dataloader):
            print("Train: ", f' {train_steps}')

            features = model.encode_image(x.to(device))
            labels = list(map(lambda label: train_dataset.label_map[label], y))

            train_features.append(features)
            train_labels.append(labels)

            # increment counter and run garbage collector
            train_steps += 1
            gc.collect()
            break

    # get testing features
    test_features = []
    test_labels = []
    with torch.no_grad():

        test_steps = 0
        for x, _ in iter(test_dataloader):
            print("Test: ", f' {test_steps}')

            features = model.encode_image(x.to(device))
            labels = list(map(lambda label: train_dataset.label_map[label], y))

            test_features.append(features)
            test_labels.append(labels)

            # increment counter and run garbage collector
            test_steps += 1
            gc.collect()
            break

    # Perform logistic regression
    classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
    classifier.fit(train_features, train_labels)

    # Evaluate using the logistic regression classifier
    predictions = classifier.predict(test_features)

    classes = test_labels #torch.cat(test_labels).cpu().numpy()

    num_correct_test_predictions_top1 = correct_predictions(classes, predictions, top_k=1)
    num_correct_test_predictions_top2 = correct_predictions(classes, predictions, top_k=2)
    num_correct_test_predictions_top3 = correct_predictions(classes, predictions, top_k=3)
    num_correct_test_predictions_top5 = correct_predictions(classes, predictions, top_k=5)

    # calculate the top-k accuracy
    test_accuracy1 = num_correct_test_predictions_top1 / test_image_count * 100
    test_accuracy2 = num_correct_test_predictions_top2 / test_image_count * 100
    test_accuracy3 = num_correct_test_predictions_top3 / test_image_count * 100
    test_accuracy5 = num_correct_test_predictions_top5 / test_image_count * 100

    print(
        "Test accuracy 1: {:.4f}, Test accuracy 2: {:.4f}, Test accuracy 3: {:.4f}, Test accuracy 5: {:.4f}".format(
             test_accuracy1, test_accuracy2, test_accuracy3, test_accuracy5
        )
    )

clip_module()


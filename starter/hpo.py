import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets

from tqdm import tqdm

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, criterion):
    logger.info("Testing started.")
    test_loss = correct = 0

    model.to("cpu")
    model.eval()

    with torch.no_grad():
        for (data, target) in test_loader:
            outputs = model(data)

            loss = criterion(outputs, target)
            test_loss += loss.item()

            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {correct / len(test_loader.dataset)}")
    logger.info("Testing completed.")


def train(model, train_loader, valid_loader, criterion, optimizer, epochs, device):
    logger.info("Training started.")
    for i in tqdm(range(epochs), desc="Training"):

        train_loss = 0
        model.train()

        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            outputs = model(data)

            loss = criterion(outputs, target)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader.dataset)
        print(f"Epoch {i}: Train loss = {train_loss:.4f}")

        val_loss = 0
        model.eval()

        running_corrects = 0
        with torch.no_grad():
            for data, target in valid_loader:
                data = data.to(device)
                target = target.to(device)

                outputs = model(data)

                loss = criterion(outputs, target)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == target.data).item()

            val_loss /= len(valid_loader.dataset)
            print(f"Epoch {i}: Val loss = {val_loss:.4f}")

        total_acc = running_corrects / len(valid_loader.dataset)
        logger.info(f"Valid average accuracy: {100 * total_acc}%")

    logger.info("Training completed.")


def net(num_classes, device):
    logger.info("Model creation for fine-tuning started.")
    model = models.efficientnet_b4(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    layer = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, num_classes, bias=False)
    )

    model.classifier[1] = layer

    model = model.to(device)
    logger.info("Model creation completed.")

    return model


def create_data_loader(data, batch_size, shuffle):
    """
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    """
    logger.info("Data loader creation started")
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    logger.info("Data loader creation completed")

    return data_loader


def create_transform(split, image_size):
    logger.info("Transformation pipeline creation started")

    pretrained_size = image_size

    if split == "train":

        train_transforms = transforms.Compose([
            transforms.Resize((pretrained_size, pretrained_size)),
            transforms.ToTensor(),
        ])

        logger.info("Transformation pipeline creation completed")
        return train_transforms

    elif split == "valid":
        valid_transforms = transforms.Compose([
            transforms.Resize((pretrained_size, pretrained_size)),
            transforms.ToTensor(),
        ])

        logger.info("Transformation pipeline creation completed")
        return valid_transforms

    elif split == "test":
        test_transforms = transforms.Compose([
            transforms.Resize((pretrained_size, pretrained_size)),
            transforms.ToTensor(),
        ])
        logger.info("Transformation pipeline creation completed")
        return test_transforms


def main(args):
    model = net(args.num_classes, args.device)

    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.classifier.parameters(), lr=args.learning_rate)

    # train_dir = os.path.join(args.data_dir, 'train')
    # valid_dir = os.path.join(args.data_dir, 'valid')
    # test_dir = os.path.join(args.data_dir, 'test')
    train_dir = os.path.join(args.train_dir)
    valid_dir = os.path.join(args.valid_dir)
    test_dir = os.path.join(args.test_dir)

    train_transform = create_transform("train", args.image_size)
    valid_transform = create_transform("valid", args.image_size)
    test_transform = create_transform("test", args.image_size)

    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)

    train_loader = create_data_loader(train_data, args.batch_size, True)
    valid_loader = create_data_loader(valid_data, args.batch_size, False)
    test_loader = create_data_loader(test_data, args.batch_size, False)

    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train(model, train_loader, valid_loader, loss_criterion, optimizer, args.epochs, args.device)

    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion)

    '''
    TODO: Save the trained model
    '''
    torch.save(
        model.cpu().state_dict(),
        os.path.join(
            args.model_path,
            "model.pth"
        )
    )
    logger.info("Model weights saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--model_path", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train_dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--valid_dir", type=str, default=os.environ["SM_CHANNEL_VALID"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    # parser.add_argument("--data_dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--image_size", type=int, default=224)

    args, _ = parser.parse_known_args()

    main(args)

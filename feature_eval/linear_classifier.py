import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder


class labeledDataset(Dataset):
    """Labeled dataset."""

    def __init__(self, root_folder, size=96):
        self.root_folder = root_folder
        self.image_folder = ImageFolder(root=root_folder)
        self.transform = transforms.Compose(
            [transforms.Resize(size=(size, size)), transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.image_folder)  # Number of images in the dataset

    def __getitem__(self, index):
        # Get a random image and its label
        img, label = self.image_folder[index]
        img = self.transform(img)

        return img, label

# Focal Loss Implementation
class FocalLoss(nn.Module):
    def __init__(self, gamma=2., alpha=0.25, reduce=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # pt is the probability of the correct class
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduce:
            return loss.mean()
        return loss


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def main():
    parser = argparse.ArgumentParser(description="Linear Classifier Training")
    parser.add_argument(
        "--train_dataset_name",
        type=str,
        required=True,
        help="name of the training dataset",
    )
    parser.add_argument(
        "--test_dataset_name", type=str, required=True, help="name of the test dataset"
    )
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument(
        "--workers", type=int, default=4, help="number of data loading workers"
    )
    parser.add_argument(
        "--arch", type=str, default="resnet18", help="architecture of the model"
    )
    parser.add_argument(
        "--out_dim", type=int, default=128, help="output dimension of the model"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="path to the checkpoint"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="number of training epochs"
    )
    parser.add_argument(
        "--num_classes", type=int, default=40, help="number of classes"
    )
    parser.add_argument(
        "--loss_fn", type=str, default="focal", help="loss function to use (focal or cross_entropy)"
    )

    args = parser.parse_args()

    # check if gpu training is available
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    train_dataset = labeledDataset(args.train_dataset_name)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    test_dataset = labeledDataset(args.test_dataset_name)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    if args.arch == "resnet18":
        model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(
            args.device
        )
    elif args.arch == "resnet50":
        model = torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes).to(
            args.device
        )

    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    state_dict = checkpoint["state_dict"]

    for k in list(state_dict.keys()):

        if k.startswith("backbone."):
            if k.startswith("backbone") and not k.startswith("backbone.fc"):
                # remove prefix
                state_dict[k[len("backbone.") :]] = state_dict[k]
        del state_dict[k]

    log = model.load_state_dict(state_dict, strict=False)
    assert log.missing_keys == ["fc.weight", "fc.bias"]

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ["fc.weight", "fc.bias"]:
            param.requires_grad = False

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0008)
    if args.loss_fn == "focal":
        criterion = FocalLoss().to(args.device)
    else:
        criterion = torch.nn.CrossEntropyLoss().to(args.device)

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter()

    for epoch in range(args.epochs):
        top1_train_accuracy = 0

        for counter, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(args.device)
            y_batch = y_batch.to(args.device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            top1 = accuracy(logits, y_batch, topk=(1,))
            top1_train_accuracy += top1[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        top1_train_accuracy /= counter + 1
        writer.add_scalar("Train/Top1_Accuracy", top1_train_accuracy.item(), epoch)

        top1_accuracy = 0
        top5_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch.to(args.device)
            y_batch = y_batch.to(args.device)

            logits = model(x_batch)

            top1, top5 = accuracy(logits, y_batch, topk=(1, 5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]

        top1_accuracy /= counter + 1
        top5_accuracy /= counter + 1
        writer.add_scalar("Test/Top1_Accuracy", top1_accuracy.item(), epoch)
        writer.add_scalar("Test/Top5_Accuracy", top5_accuracy.item(), epoch)

        print(
            f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}"
        )

    writer.close()


if __name__ == "__main__":
    main()

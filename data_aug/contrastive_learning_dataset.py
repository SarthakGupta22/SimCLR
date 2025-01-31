# from torchvision.transforms import transforms
# from data_aug.gaussian_blur import GaussianBlur
# from torchvision import transforms, datasets
# from data_aug.view_generator import ContrastiveLearningViewGenerator
# from exceptions.exceptions import InvalidDatasetSelection


# class ContrastiveLearningDataset:
#     def __init__(self, root_folder):
#         self.root_folder = root_folder

#     @staticmethod
#     def get_simclr_pipeline_transform(size, s=1):
#         """Return a set of data augmentation transformations as described in the SimCLR paper."""
#         color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
#         data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
#                                               transforms.RandomHorizontalFlip(),
#                                               transforms.RandomApply([color_jitter], p=0.8),
#                                               transforms.RandomGrayscale(p=0.2),
#                                               GaussianBlur(kernel_size=int(0.1 * size)),
#                                               transforms.ToTensor()])
#         return data_transforms

#     def get_dataset(self, name, n_views):
#         valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
#                                                               transform=ContrastiveLearningViewGenerator(
#                                                                   self.get_simclr_pipeline_transform(32),
#                                                                   n_views),
#                                                               download=True),

#                           'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
#                                                           transform=ContrastiveLearningViewGenerator(
#                                                               self.get_simclr_pipeline_transform(96),
#                                                               n_views),
#                                                           download=True)}

#         try:
#             dataset_fn = valid_datasets[name]
#         except KeyError:
#             raise InvalidDatasetSelection()
#         else:
#             return dataset_fn()


import os
import random

from data_aug.gaussian_blur import GaussianBlur
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.stl10 import STL10


class RandomRotation90_180_270WithProbability(transforms.RandomRotation):
    def __init__(self, probability=0.5):
        """
        Args:
            probability: Probability of applying the random rotation (default is 0.5)
        """
        self.probability = probability

    def __call__(self, img):
        # Apply the rotation with a certain probability
        if random.random() < self.probability:
            # Choose a random angle from the set {90, 180, 270}
            angle = random.choice([90, 180, 270])
            return transforms.functional.rotate(img, angle)
        else:
            # Return the original image if no rotation is applied
            return img


class ContrastiveLearningDataset:
    def __init__(self, root_folder, mode="unlabeled", batch_size=32, n_views=2):
        """
        Args:
            root_folder: root folder where datasets are stored.
            mode: Defines the data loading strategy ('unlabeled' or 'contrastive').
            batch_size: Batch size for the DataLoader.
            n_views: Number of views for contrastive learning.
        """
        self.root_folder = root_folder
        self.mode = mode
        self.batch_size = batch_size
        self.n_views = n_views

    def get_simclr_pipeline_transform(self, size, schematic_data=False, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        if not schematic_data:
            transforms_list = [
                transforms.RandomResizedCrop(size=size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(kernel_size=int(0.1 * size)),
                transforms.ToTensor(),
            ]
        else:
            transforms_list = [
                transforms.Resize(size=(size, size)),
                RandomRotation90_180_270WithProbability(probability=0.5),
                transforms.RandomApply([color_jitter], p=0.4),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(
                    kernel_size=int(0.1 * size)
                ),  # Gaussian Blur here can be useful if images are very small, it can smoothen it out
                transforms.ToTensor(),
            ]

        data_transforms = transforms.Compose(transforms_list)
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {
            "cifar10": lambda: datasets.CIFAR10(
                self.root_folder,
                train=True,
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(32), n_views
                ),
                download=True,
            ),
            "stl10": lambda: datasets.STL10(
                self.root_folder,
                split="unlabeled",
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(96), n_views
                ),
                download=True,
            ),
            "ff_devices_unlabeled": lambda: UnlabeledContrastiveDataset(
                self.root_folder,
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(96, schematic_data=True), n_views
                ),
                # n_views=n_views,
            ),  # Unlabeled data: Randomly load data from any class
            "ff_devices_labeled": lambda: ContrastiveLearningDatasetWithAugmentations(
                self.root_folder,
                transform=self.get_simclr_pipeline_transform(96, schematic_data=True),
                n_views=n_views,
                size=96,
            ),
            "ff_devices_one_shot": lambda: OneShotContrastiveLearningDataset(
                self.root_folder,
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(96, schematic_data=True), n_views
                ),
            ),  # One-shot contrastive learning data: One image per class as the support image
        }  # Contrastive learning data: Data loaded in pairs from the same class

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()

    # def get_dataloader(self):
    #     """Return DataLoader for the selected dataset."""
    #     return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)


class UnlabeledContrastiveDataset(Dataset):
    """Dataset where data is treated as unlabeled and loaded randomly from any class."""

    def __init__(self, root_folder, transform):
        self.root_folder = root_folder
        self.transform = transform
        self.image_folder = ImageFolder(root=root_folder)

        self.classes = self.image_folder.classes

    def __len__(self):
        return len(self.image_folder)  # Number of images in the dataset

    def __getitem__(self, index):
        # Get a random image and its transformation
        img, label = self.image_folder[index]
        img = self.transform(img)

        return img, label


class ContrastiveLearningDatasetWithAugmentations(Dataset):
    """Dataset where each sample is a positive pair, i.e., images from the same class."""

    def __init__(self, root_folder, transform, n_views=2, size=96):
        self.root_folder = root_folder
        self.transform = transform
        self.n_views = n_views
        self.size = size

        # Create a custom ImageFolder-style dataset
        self.image_folder = ImageFolder(root=root_folder)
        self.classes = self.image_folder.classes
        self.class_to_idx = self.image_folder.class_to_idx

        # Group images by class
        self.class_images = {class_idx: [] for class_idx in range(len(self.classes))}
        for idx, (img, label) in enumerate(self.image_folder):
            self.class_images[label].append(idx)

    def __len__(self):
        return len(self.image_folder)  # Number of images in the dataset

    def __getitem__(self, index):
        # Get the image and its label
        img, label = self.image_folder[index]

        # Select a random image from the same class to create a positive pair
        positive_idx = random.choice(self.class_images[label])

        # Ensure that we don't pick the same image as the positive pair
        while positive_idx == index:
            positive_idx = random.choice(self.class_images[label])

        # Get the positive pair image
        positive_img, _ = self.image_folder[positive_idx]

        # Apply transformations to both images (the query and the positive pair) with p probability
        p = 0.5
        if random.random() < p:
            img = self.transform(img)
        else:
            img = transforms.Compose(
                [transforms.Resize((self.size, self.size)), transforms.ToTensor()]
            )(img)
        if random.random() < p:
            positive_img = self.transform(positive_img)
        else:
            positive_img = transforms.Compose(
                [transforms.Resize((self.size, self.size)), transforms.ToTensor()]
            )(positive_img)

        views = [img, positive_img]  # Return the original images without transformation
        return views, label


class OneShotContrastiveLearningDataset(Dataset):
    """One-shot contrastive learning dataset: One image per class as the support image, augmented into n_views."""

    def __init__(self, root_folder, transform):
        self.root_folder = root_folder
        self.transform = transform

        # Create a custom ImageFolder-style dataset
        self.image_folder = ImageFolder(root=root_folder)
        self.classes = self.image_folder.classes
        self.class_to_idx = self.image_folder.class_to_idx

        # Preselect one image from each class for the support set
        self.support_images = {}

        for class_idx in range(len(self.classes)):
            # Get all indices of images belonging to the current class
            class_indices = [
                idx
                for idx, (img, label) in enumerate(self.image_folder)
                if label == class_idx
            ]
            # Randomly select one image from this class as the support image
            support_idx = random.choice(class_indices)
            self.support_images[class_idx] = (
                support_idx  # Store the index of the support image
            )

    def __len__(self):
        return len(self.classes)  # One image per class

    def __getitem__(self, index):
        # Get the support image for this class
        support_idx = self.support_images[index]
        support_img, label = self.image_folder[support_idx]

        # Apply the transformation to the support image to generate `n_views` augmentations
        views = self.transform(support_img)

        return views, label

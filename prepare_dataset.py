import os
import torch
import torchvision
from PIL import Image
from tqdm import tqdm

def prepare_cifar100(output_path='./datasets'):
    """
    Downloads the CIFAR-100 dataset and saves it in the ImageFolder format.
    """
    # Define the root path for the final structured dataset
    cifar100_path = os.path.join(output_path, 'cifar100_images')

    # Create directories for train and validation sets
    train_dir = os.path.join(cifar100_path, 'train')
    val_dir = os.path.join(cifar100_path, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    print("Downloading and preparing CIFAR-100 dataset...")

    # --- Process the Training Set ---
    train_dataset = torchvision.datasets.CIFAR100(
        root=output_path, train=True, download=True
    )
    print("\nProcessing training images...")
    for i, (image, label) in enumerate(tqdm(train_dataset)):
        class_name = train_dataset.classes[label]
        class_dir = os.path.join(train_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        # image is a PIL Image object, so we can save it directly
        image.save(os.path.join(class_dir, f'train_{i}.png'))

    # --- Process the Test Set (as validation) ---
    test_dataset = torchvision.datasets.CIFAR100(
        root=output_path, train=False, download=True
    )
    print("\nProcessing validation (test) images...")
    for i, (image, label) in enumerate(tqdm(test_dataset)):
        class_name = test_dataset.classes[label]
        class_dir = os.path.join(val_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        image.save(os.path.join(class_dir, f'val_{i}.png'))

    print(f"\nDataset successfully prepared at: {cifar100_path}")


if __name__ == '__main__':
    # By default, this will create a './datasets/cifar100_images' directory
    prepare_cifar100()
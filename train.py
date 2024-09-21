import torch
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.amp
from model import Unet
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    visualize_rgb
)


def define_transforms():
    train_transform = A.Compose([
        # A.Resize(256, 256),  # Reduced from 513x513 to 256x256
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    val_transforms = A.Compose([
        # A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return train_transform, val_transforms


def train_fn(model, dataloader, optimizer, loss_fn, device, epoch, num_epochs):
    model.train()
    loop = tqdm(dataloader, leave=True)
    for images, targets in loop:
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1).float()  # Ensure targets are float

        optimizer.zero_grad()
        output = model(images)['out']

        if output.dtype != torch.float32:
            output = output.float()

        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
        loop.set_postfix(loss=loss.item())


def main():
    LOAD_MODEL = False
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {DEVICE}")

    model = Unet(output_channels=1, in_channels=12).to(
        DEVICE)  # Assuming you are using multi-spectral data with 12 channels
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()  # Ensure that targets are appropriately shaped

    scaler = torch.amp.GradScaler() if DEVICE == 'cuda' else None

    train_transform, val_transforms = define_transforms()
    train_loader, val_loader = get_loaders(
        "data/train_images/", "data/train_masks/",
        "data/val_images/", "data/val_masks/",
        8,
        train_transform, val_transforms,
        num_workers=2,
        pin_memory=False
    )

    first_batch_images, first_batch_labels = next(iter(train_loader))
    visualize_rgb(first_batch_images, index=0)  #

    first_batch_images, first_batch_labels = next(iter(train_loader))
    visualize_rgb(first_batch_images, index=0)  # You can change index to visualize different images in the batch

    if LOAD_MODEL and os.path.isfile("my_checkpoint.pth.tar"):
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    num_epochs = 3 # Define this before the loop
    for epoch in range(num_epochs):
        train_fn(model, train_loader, optimizer, loss_fn, DEVICE, epoch, num_epochs)
        check_accuracy(val_loader, model, device=DEVICE)

    save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()})
    save_predictions_as_imgs(val_loader, model, folder="saved_images", device=DEVICE)


if __name__ == "__main__":
    main()

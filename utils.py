import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from PIL import Image
from dataset import SatelliteDataset


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(train_dir, train_maskdir, val_dir, val_maskdir, batch_size, train_transform, val_transform,
                num_workers=2, pin_memory=False):
    train_ds = SatelliteDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                              shuffle=True)

    val_ds = SatelliteDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                            shuffle=False)

    return train_loader, val_loader




def calculate_iou(preds, targets, threshold=0.3):
    """ Calculate Intersection over Union (IoU) for water class (class = 1). """
    preds = (preds > threshold).float()  # Convert predictions to binary (water class vs background)
    intersection = (preds * targets).sum()
    union = (preds + targets).sum() - intersection
    iou = intersection / (union + 1e-8)  # Add epsilon to avoid division by zero
    return iou.item()


def calculate_precision(preds, targets, threshold=0.5):
    """ Calculate Precision for water class (class = 1). """
    preds = (preds > threshold).float()
    true_positives = (preds * targets).sum()
    predicted_positives = preds.sum()
    precision = true_positives / (predicted_positives + 1e-8)
    return precision.item()


def calculate_recall(preds, targets, threshold=0.5):
    """ Calculate Recall for water class (class = 1). """
    preds = (preds > threshold).float()
    true_positives = (preds * targets).sum()
    actual_positives = targets.sum()
    recall = true_positives / (actual_positives + 1e-8)
    return recall.item()


def calculate_f1(precision, recall):
    """ Calculate F1-score from precision and recall. """
    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def visualize_rgb(image_tensor, index=0, folder="rgb_visualization", file_prefix="image", red_band=3, green_band=2,
                  blue_band=1):
    """ Visualize the multispectral image as an RGB image by combining specified bands from a batched tensor. """
    os.makedirs(folder, exist_ok=True)  # Ensure directory exists

    # Check if the tensor is on CPU or move it to CPU
    if image_tensor.is_cuda:
        image_tensor = image_tensor.cpu()

    # Extract the RGB bands from the specified image in the batch
    red = image_tensor[index, red_band, :, :].numpy()
    green = image_tensor[index, green_band, :, :].numpy()
    blue = image_tensor[index, blue_band, :, :].numpy()

    # Normalize the bands to the [0, 1] range for visualization (if necessary)
    red = (red - red.min()) / (red.max() - red.min())
    green = (green - green.min()) / (green.max() - green.min())
    blue = (blue - blue.min()) / (blue.max() - blue.min())

    # Stack the channels to create an RGB image
    rgb_image = np.stack([red, green, blue], axis=-1)

    # Plot and save the RGB image
    plt.figure()
    plt.imshow(rgb_image)
    plt.title("RGB Visualization")
    plt.axis('off')  # Turn off axis labels for clean visualization

    file_path = os.path.join(folder, f"{file_prefix}_{index}_rgb.png")
    plt.savefig(file_path)
    plt.close()

    print(f"RGB image visualized and saved at {file_path}")


def check_accuracy(loader, model, is_validation=True, device="cuda"):
    model.eval()  # Always set the model to evaluation mode for accuracy check

    num_correct = 0
    num_pixels = 0
    dice_score = 0
    total_iou = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    count = 0  # To track the number of batches for averaging

    with torch.no_grad():  # Disable gradient calculation
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)  # Ensure mask has the same shape as the output

            # Get model predictions
            outputs = model(x)
            preds = outputs['out']  # Accessing the primary output
            preds = torch.sigmoid(preds)  # Apply sigmoid to obtain probabilities
            preds = (preds > 0.5).float()  # Convert probabilities to binary output

            # Calculate the number of correct pixels
            num_correct += (preds == y).sum().item()
            num_pixels += torch.numel(preds)

            # Calculate the Dice score
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

            # Compute metrics for each batch
            iou = calculate_iou(preds, y)
            precision = calculate_precision(preds, y)
            recall = calculate_recall(preds, y)
            f1 = calculate_f1(precision, recall)

            # Update totals for averaging later
            total_iou += iou
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            count += 1

    # Calculate average metrics across all batches
    avg_iou = total_iou / count
    avg_precision = total_precision / count
    avg_recall = total_recall / count
    avg_f1 = total_f1 / count

    # Calculate accuracy as a percentage
    accuracy = num_correct / num_pixels * 100

    print(f"Got {num_correct}/{num_pixels} correct with accuracy {accuracy:.2f}%")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")

    model.train()  # Return the model to training mode if it's not the validation set


def save_tensor_as_image(tensor, folder, filename):
    """Saves a tensor as an image."""
    tensor = tensor.to('cpu')
    grid = make_grid(tensor)

    # Scale to [0, 255], round off to the nearest integer, and clamp
    ndarr = grid.mul(255).add(0.5).clamp(0, 255)
    ndarr = ndarr.to(torch.uint8)  # Convert to uint8 only after all modifications

    # Permute dimensions to match image format
    ndarr = ndarr.permute(1, 2, 0)

    # Handle potential issues with shape, e.g., single channel grayscale images
    if ndarr.shape[2] == 1:
        ndarr = ndarr.squeeze(2)  # If only one channel, remove the channel dimension

    im = Image.fromarray(ndarr.numpy(), 'L' if ndarr.ndim == 2 else 'RGB')
    os.makedirs(folder, exist_ok=True)  # Ensure the directory exists
    im.save(os.path.join(folder, f'{filename}.png'))
    print(f"Saved image to {os.path.join(folder, f'{filename}.png')}")


def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda", fileName=""):
    model.eval()  # Set model to evaluation mode
    for idx, (x, y) in enumerate(loader):
        # x = x.to(device)

        with torch.no_grad():
            outputs = model(x)  # Model outputs are expected to be a dictionaryلا
            preds = outputs['out']  # Extract the tensor associated with the key 'out'

            preds = torch.sigmoid(preds)  # Apply sigmoid to convert to probabilities
            preds = (preds > 0.5).float()  # Threshold to get binary mask
            preds = (preds * 255).byte()  # Scale to 0-255 and convert to uint8 for visualization or storage

        # Ensure the folder exists before trying to save files
        os.makedirs(folder, exist_ok=True)
        if fileName == "":
            fileName = idx
        save_tensor_as_image(preds, folder, f"pred_{fileName}")  # Save the predictions

        # Adjust and save ground truth images if necessary
        if y.dim() == 4:
            y = y.squeeze(1)  # Reduce it to [batch, height, width] if necessary
        save_tensor_as_image(y * 255, folder, f"y_{fileName}")  # Scale ground truth for consistency with predictions

    model.train()

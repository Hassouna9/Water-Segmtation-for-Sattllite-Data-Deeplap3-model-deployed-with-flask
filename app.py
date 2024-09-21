from collections import OrderedDict
from model import Unet
from flask import Flask, request, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import os
import numpy as np
from PIL import Image
from rasterio.io import MemoryFile
from torchvision.utils import make_grid

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Configuration
UPLOAD_FOLDER = 'data/uploads'
OUTPUT_FOLDER = 'static/output_images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_rgb_image(file, OUTPUT_FOLDER, fileName):
    """
    Extracts RGB channels from a 12-channel .tif image and saves the RGB visualization.

    Args:
        file (FileStorage): The uploaded .tif file object from Flask.
        OUTPUT_FOLDER (str): Directory where the RGB image will be saved.
        fileName (str): Base filename for the saved RGB image.
    """
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    with MemoryFile(file.read()) as memfile:
        with memfile.open() as dataset:
            img = dataset.read()  # Shape: (12, H, W)

    red_band = 3
    green_band = 2
    blue_band = 1

    rgb = img[[red_band, green_band, blue_band], :, :]  # Shape: (3, H, W)

    rgb_min = rgb.min(axis=(1, 2), keepdims=True)
    rgb_max = rgb.max(axis=(1, 2), keepdims=True)

    rgb_max[rgb_max == rgb_min] = rgb_min[rgb_max == rgb_min] + 1e-6
    rgb_normalized = ((rgb - rgb_min) / (rgb_max - rgb_min) * 255).astype(np.uint8)

    rgb_image = np.transpose(rgb_normalized, (1, 2, 0))

    rgb_pil = Image.fromarray(rgb_image)
    rgb_output_path = os.path.join(OUTPUT_FOLDER, f"{fileName}_RGB.png")
    rgb_pil.save(rgb_output_path)


def save_prediction_image(prediction, folder, filename):
    """
    Saves the prediction tensor as an image following the exact logic of save_tensor_as_image.

    Args:
        prediction (torch.Tensor or OrderedDict): The raw output from the segmentation model.
        folder (str): Directory where the prediction image will be saved.
        filename (str): Base filename for the saved prediction image.
    """
    # Ensure the directory exists
    os.makedirs(folder, exist_ok=True)

    # Extract 'out' tensor if prediction is an OrderedDict
    if isinstance(prediction, dict) or isinstance(prediction, OrderedDict):
        if 'out' in prediction:
            prediction = prediction['out']
        else:
            raise KeyError("The model's output dictionary does not contain the key 'out'.")

    # Ensure prediction is a tensor
    if not isinstance(prediction, torch.Tensor):
        raise TypeError(f"Expected prediction to be a torch.Tensor, but got {type(prediction)}")

    # Move tensor to CPU
    prediction = prediction.to('cpu')
    grid = make_grid(prediction)
    ndarr = grid.mul(255).add(0.5).clamp(0, 255)
    ndarr = ndarr.to(torch.uint8)
    ndarr = ndarr.permute(1, 2, 0)

    if ndarr.shape[2] == 1:
        ndarr = ndarr.squeeze(2)

    mode = 'L' if ndarr.ndim == 2 else 'RGB'

    im = Image.fromarray(ndarr.numpy(), mode)

    image_path = os.path.join(folder, f'{filename}_Prediction.png')
    im.save(image_path)
    print(f"Saved image to {image_path}")


model = Unet(output_channels=1, in_channels=12).to(DEVICE)

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])


def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')  # Correct usage


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            base_filename = os.path.splitext(filename)[0]

            save_rgb_image(file, OUTPUT_FOLDER, base_filename)

            file.seek(0)

            with MemoryFile(file.read()) as memfile:
                with memfile.open() as dataset:
                    img = dataset.read()  # Shape: (12, H, W)


            img_hwc = np.transpose(img, (1, 2, 0))  # Convert to (H, W, C)
            Photo_transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
            transformed = Photo_transform(image=img_hwc)
            input_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension

            # Perform segmentation
            model.eval()
            with torch.no_grad():
                output = model(input_tensor)

            # Save the prediction visualization
            save_prediction_image(output, OUTPUT_FOLDER, base_filename)

            # Define filenames for rendering
            rgb_filename = f"{base_filename}_RGB.png"
            pred_filename = f"{base_filename}_Prediction.png"

            flash('File successfully uploaded and processed')
            return render_template('output.html', uploaded_image=rgb_filename, segmented_image=pred_filename)

        else:
            flash('Allowed file types are tif, tiff')
            return redirect(request.url)

    return render_template('index.html')


if __name__ == '__main__':
    checkpoint_path = r"my_checkpoint.pth.tar"

    # Ensure the path is correct
    load_checkpoint(torch.load(checkpoint_path))
    app.run(debug=True)
    # Define device

from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
class SatelliteDataset(Dataset):

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.tif')]

    def open_tiff(self, path):
        import rasterio  # Ensure rasterio is imported here
        with rasterio.open(path) as src:
            img = src.read()
            return np.moveaxis(img, 0, -1)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.tif', '.png'))

        image = self.open_tiff(img_path)
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            transformed = self.transform(image=image, mask=np.array(mask))
            image = transformed['image']
            mask = transformed['mask']

        return image, mask

    def __len__(self):
        return len(self.images)
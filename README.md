# **DeepLab3-Based Water Segmentation for Satellite Images**

## **Project Overview**
This project leverages a DeepLabV3 model with a ResNet-50 backbone to perform semantic segmentation on satellite imagery, specifically detecting and segmenting water bodies. Built with PyTorch, the model processes multispectral images with 12 channels, making it suitable for handling diverse satellite datasets. This repository includes data preparation, model training, evaluation, and visualization tools to streamline the segmentation pipeline.

---

## **Features**
- **Multispectral Support**: Handles input with 12 channels for enhanced segmentation accuracy on satellite data.
- **Customizable DeepLabV3**: Adjusted for custom input and output configurations.
- **Real-Time Progress Tracking**: Includes progress bars and live loss updates during training.
- **Performance Metrics**: IoU, Precision, Recall, and F1 Score for comprehensive evaluation.
- **Visualization Tools**: View RGB composites of images alongside their segmentation masks.

---

## **Data Preparation**
### **Dataset**
The dataset comprises multispectral satellite images and their corresponding water segmentation masks:
- **Image Format**: `.tif` files with 12 spectral bands.
- **Mask Format**: `.png` files with binary values (1 for water, 0 for non-water).

### **Image Transformations**
Transformations ensure consistency and compatibility with the model:
- **Normalization**: Applied per ImageNet statistics \((\mu=[0.485, 0.456, 0.406], \sigma=[0.229, 0.224, 0.225])\).
- **Tensor Conversion**: Converts images to PyTorch tensors for GPU compatibility.
- **Optional Resizing**: Images can be resized to \(256 \times 256\) for computational efficiency.

---

## **Model Architecture**
### **DeepLabV3 with ResNet-50 Backbone**
- **Input Channels**: Modified to accept 12-channel multispectral input.
- **Output Channels**: Configured for binary segmentation.
- **Auxiliary Classifier**: Retained for improved training efficiency.

### **Training Details**
- **Loss Function**: Binary Cross-Entropy with Logits (BCEWithLogitsLoss).
- **Optimizer**: Adam optimizer with a learning rate of \(1 \times 10^{-3}\).
- **Mixed Precision Training**: Enabled for faster and memory-efficient training on GPUs.
- **Batch Size**: 8.
- **Epochs**: 5 (configurable).

---

## **Training Process**
### **Pipeline**
1. Load data using custom DataLoaders for training and validation datasets.
2. Train the model using:
   - Forward pass for prediction.
   - Loss computation and backward pass for parameter updates.
3. Evaluate after each epoch using metrics like accuracy, IoU, Precision, Recall, and F1 Score.

### **Metrics**
| **Metric**   | **Definition**                                              |
|--------------|-------------------------------------------------------------|
| **IoU**      | Measures overlap between predictions and ground truth masks. |
| **Precision**| Fraction of correctly predicted water pixels.               |
| **Recall**   | Fraction of actual water pixels correctly identified.       |
| **F1 Score** | Harmonic mean of Precision and Recall.                      |

### **Training Results**
| **Epoch** | **Accuracy** | **Average IoU** | **Precision** | **Recall** | **F1 Score** |
|-----------|--------------|-----------------|---------------|------------|--------------|
| 1         | 85.85%       | 0.5654          | 0.8246        | 0.6659     | 0.7009       |
| 2         | 86.04%       | 0.6318          | 0.7225        | 0.8359     | 0.7711       |
| 3         | 81.50%       | 0.5435          | 0.6814        | 0.7114     | 0.6882       |
| 4         | 87.61%       | 0.6302          | 0.8535        | 0.6994     | 0.7573       |
| 5         | *To be continued*... |

---

## **Visualization**
The visualization tool generates side-by-side comparisons of:
- **RGB Images**: Composed from specified spectral bands.
- **Segmentation Masks**: Binary masks highlighting water bodies.

Command for visualization:
```python
visualize_image_and_mask(images, masks, indices=[0, 1, 2])
```
---

### **How to Use**

You can utilize this project in two ways:

1. **Google Colab Notebook**:  
   - Open the `Segmentation.ipynb` file in Google Colab for an interactive environment.
   - It includes all the steps for data preparation, model training, evaluation, and visualization.

2. **Local Python Code**:  
   - Clone the repository to your local machine:
     ```bash
     git clone https://github.com/yourusername/WaterSegmentation.git
     ```

   - Use the provided Python scripts:
     - **`train.py`**: For training the model.
     - **`app.py`**: For deploying or visualizing results.
     - **`dataset.py`, `model.py`, `utils.py`**: Helper scripts for dataset preparation, model configuration, and utilities.

---

## **Future Improvements**
- Add support for additional classes beyond binary water segmentation.
- Explore advanced data augmentation for improved generalization.
- Optimize model architecture for faster inference on edge devices.

---

## **Acknowledgments**
This project uses the DeepLabV3 implementation from PyTorch and depends on libraries such as `torchvision`, `rasterio`, and `albumentations`.

--- 


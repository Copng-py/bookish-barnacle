## MNIST Image Classification using CNN
This repository contains a PyTorch implementation of a Convolutional Neural Network (CNN) model for classifying MNIST handwritten digits.
<p align="center">
  <img src="https://github.com/Copng-py/bookish-barnacle/assets/146678457/c0fafdb2-53d4-442a-b65d-9eaaab76a7fb" width="500" height="500">
</p>

### Model Architecture
The CNN model used in this project consists of the following layers:
- **Convolutional Layer 1**: 16 filters, 5x5 kernel size, ReLU activation
- **Max Pooling Layer 1**: 2x2 kernel size
- **Convolutional Layer 2**: 32 filters, 3x3 kernel size, ReLU activation
- **Max Pooling Layer 2**: 2x2 kernel size
- **Convolutional Layer 3**: 16 filters, 1x1 kernel size, ReLU activation
- **Fully Connected Layer 1**: 64 units, ReLU activation
- **Fully Connected Layer 2 (Output)**: 10 units (corresponding to 10 digit classes)



### Data Processing
The MNIST dataset is used for this project. The data is preprocessed as follows:

- **Grayscale Conversion:** The input images are converted to grayscale (1 channel).
- **Resizing:** The images are resized to 28x28 pixels.
- **Normalization:** The pixel values are converted to tensors and normalized to the range [0, 1].

### Training Function
The train_model() function is responsible for training the CNN model. It includes the following key components:

- **Device:** The model is moved to the available GPU device (if CUDA is available) or CPU.
- **Loss Function:** The CrossEntropyLoss is used as the loss function.
- **Optimizer:** The Adam optimizer is used with a learning rate of 0.001.
- **Learning Rate Scheduler:** A MultiStepLR scheduler is used to decrease the learning rate by a factor of 0.01 at epochs 5 and 8.
- **Training Loop:** The model is trained for 10 epochs, with the training loss and test accuracy being calculated and reported at the end of each epoch.

### Learning Rate and Model Performance
- **Learning Rate:** The initial learning rate is set to 0.001, and the MultiStepLR scheduler is used to reduce the learning rate during training.
- **Model Performance:** The final test accuracy achieved by the model is 99.17%.
![image](https://github.com/Copng-py/bookish-barnacle/assets/146678457/4aa51d11-d398-40c3-bc15-5a8d1be0bae9)
<img width="988" alt="Screenshot 2024-05-11 at 2 46 14 PM" src="https://github.com/Copng-py/bookish-barnacle/assets/146678457/4b6e3265-afd5-44a5-bbd5-115eb20477a4">

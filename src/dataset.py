import matplotlib.pyplot as plt
import numpy as np

import torch
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from torchvision.transforms import ToTensor, Resize, Grayscale

def load_mnist():
    transform = transforms.Compose([
        Grayscale(num_output_channels=1),  # Convert to grayscale
        Resize((28, 28)),                   # Resize to (28, 28)
        ToTensor()                          # Convert to tensor
    ])
    
    trainset = MNIST(root='./data', train=True, download=True, transform=transform)
    testset = MNIST(root='./data', train=False, download=True, transform=transform)
    return trainset, testset



def load_cifar10():
    trainset = CIFAR10(root='./data', train=True,
                       download=True, transform=ToTensor())
    testset = CIFAR10(root='./data', train=False,
                      download=True, transform=ToTensor())
    return trainset, testset


def load_fashion_mnist():
    trainset = FashionMNIST(root='./data', train=True,
                       download=True, transform=ToTensor())
    testset = FashionMNIST(root='./data', train=False,
                      download=True, transform=ToTensor())
    return trainset, testset


# functions to show an image
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

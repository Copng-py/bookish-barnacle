import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from model import CNNModel
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define a function to show an image
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Define a function to load the MNIST dataset with preprocessing
def load_mnist():
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((28, 28)),                  # Resize to (28, 28)
        transforms.ToTensor()                         # Convert to tensor
    ])
    
    trainset = MNIST(root='./data', train=True, download=True, transform=transform)
    testset = MNIST(root='./data', train=False, download=True, transform=transform)
    return trainset, testset

# Define the training function
def train_model(model, trainloader, testloader, num_epochs=10, learning_rate=0.001):
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Use Adam optimizer
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 8], gamma=0.01)
    
    train_losses = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 1000 == 999:    # Print every 1000 mini-batches
                print('[Epoch %d, Mini-batch %5d] Loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                print('Learning Rate: %.6f' % optimizer.param_groups[0]['lr'])
                running_loss = 0.0
        
        # Update the learning rate
        scheduler.step()
        
        # Calculate training accuracy
        train_acc = accuracy(model, trainloader, device)
        print('Epoch %d: Training Accuracy: %.2f%%' % (epoch + 1, train_acc * 100))
        
        # Calculate test accuracy
        test_acc = accuracy(model, testloader, device)
        print('Epoch %d: Test Accuracy: %.2f%%' % (epoch + 1, test_acc * 100))
        
        train_losses.append(running_loss)
        test_accuracies.append(test_acc)
    
    print('Finished Training')
    return train_losses, test_accuracies

# Define a function to calculate accuracy
def accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Load the MNIST dataset with preprocessing
trainset, testset = load_mnist()
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Initialize the model
model = CNNModel()

# Train the model
train_losses, test_accuracies = train_model(model, trainloader, testloader, num_epochs=10, learning_rate=0.001)

# Save the trained model
torch.save(model.state_dict(), 'model.pth')

# Display a sample prediction
dataiter = iter(testloader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % labels[j].item() for j in range(len(labels))))

images = images.to(device)
outputs = model(images)
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % predicted[j].item() for j in range(len(predicted))))

# Calculate and print final test accuracy
final_test_acc = accuracy(model, testloader, device)
print('Final Test Accuracy: %.2f%%' % (final_test_acc * 100))

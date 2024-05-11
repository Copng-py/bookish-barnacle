import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # convolutional layer 1
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5),
            nn.ReLU(),
        )
        
        # max pooling layer 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        
        # convolutional layer 2
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
        )
        
        # max pooling layer 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        # convolutional layer 3
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1),
            nn.ReLU(),
        )
        
        # fully connected layers
        # Calculate the size of the input to the first FC layer after convolutions and pooling
        self.fc_input_size = self._get_fc_input_size()
        
        self.fc_layer1 = nn.Sequential(
            nn.Linear(in_features=self.fc_input_size, out_features=64),
            nn.ReLU(),
        )
        
        # Output layer (fully connected)
        self.fc_layer2 = nn.Linear(in_features=64, out_features=10)
        
    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.maxpool1(x)
        x = self.conv_layer2(x)
        x = self.maxpool2(x)
        x = self.conv_layer3(x)
        
        # Flatten the output from convolutional layers
        x = x.view(x.size(0), -1)
        
        x = self.fc_layer1(x)
        x = self.fc_layer2(x)
        
        return x
    
    def _get_fc_input_size(self):
        # Create a sample input and forward it through the layers to get the output size
        sample_input = torch.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
        sample_output = self.conv_layer1(sample_input)
        sample_output = self.maxpool1(sample_output)
        sample_output = self.conv_layer2(sample_output)
        sample_output = self.maxpool2(sample_output)
        sample_output = self.conv_layer3(sample_output)
        
        # Flatten the output to calculate the input size for the first FC layer
        fc_input_size = sample_output.view(sample_output.size(0), -1).size(1)
        
        return fc_input_size

# Instantiate the model
cnn_model = CNNModel()
print(cnn_model)

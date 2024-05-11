import torch
from torchviz import make_dot
from model import CNNModel

# Create a dummy input that corresponds to the input size of the model
dummy_input = torch.randn(1, 1, 28, 28)

# Instantiate your model
model = CNNModel()

# Perform a forward pass (to get the output logit)
output = model(dummy_input)

# Create the visual graph
dot = make_dot(output, params=dict(list(model.named_parameters()) + [('input', dummy_input)]))

# Render and save the visualization as a PNG file
dot.render('cnn_model_architecture', format='png', directory='visuals')
print("Diagram saved to 'visuals/cnn_model_architecture.png'")
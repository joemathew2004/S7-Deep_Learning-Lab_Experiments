import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# List of configurations to try: each tuple contains hidden layer sizes and activation function
configs = [
    ((512, 256, 128), 'relu'),
    ((512, 256, 128), 'tanh'),
    ((512, 256, 128), 'sigmoid'),
    ((256, 128, 64), 'relu'),
    ((256, 128, 64), 'tanh'),
    ((256, 128, 64), 'sigmoid'),
    ((1024, 512, 256), 'relu'),
    ((1024, 512, 256), 'tanh'),
    ((1024, 512, 256), 'sigmoid')
]

# Hyperparameters
epochs = 5                 # Number of training epochs for each config
batch_size = 128           # Mini-batch size for training/testing
lr = 0.001                 # Learning rate for Adam optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

# Data preprocessing and normalization for CIFAR-10 images (3 color channels)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values to range [-1, 1]
])

# Download and load CIFAR-10 training and test datasets with transformations
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create data loaders to efficiently feed batches during training/testing
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

classes = trainset.classes  # Class names in CIFAR-10 (e.g., airplane, automobile, etc.)

# Helper function to get activation function based on string name
def get_activation(name):
    return {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid()}[name]

# Define Feedforward Neural Network with 3 hidden layers and specified activation
class FeedforwardNet(nn.Module):
    def __init__(self, hidden_units, activation):
        super().__init__()
        # Define fully connected layers; input size = 32*32*3 (image flattened)
        self.fc1 = nn.Linear(32*32*3, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], hidden_units[2])
        self.fc4 = nn.Linear(hidden_units[2], 10)  # Output layer for 10 classes
        self.act = get_activation(activation)     # Select activation function

    def forward(self, x):
        x = x.view(-1, 32*32*3)      # Flatten the image tensor
        x = self.act(self.fc1(x))    # Hidden layer 1 + activation
        x = self.act(self.fc2(x))    # Hidden layer 2 + activation
        x = self.act(self.fc3(x))    # Hidden layer 3 + activation
        x = self.fc4(x)              # Output layer (logits for each class)
        return x

# Variables to track the best performing model and its configuration
best_accuracy = 0.0
best_model = None
best_config = None

# Loop over each config, train the model, and evaluate performance
for idx, (hidden_layers, activation_function) in enumerate(configs, start=1):
    print(f"\n=== Run {idx} | Hidden: {hidden_layers}, Activation: {activation_function} ===")
   
    torch.cuda.empty_cache()  # Clear GPU memory before new run (optional)
   
    # Initialize model with current configuration and move to device
    model = FeedforwardNet(hidden_units=hidden_layers, activation=activation_function).to(device)
   
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()           # Set model to training mode
        total_loss = 0

        # Iterate over batches of training data
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)  # Move data to GPU/CPU
           
            optimizer.zero_grad()             # Reset gradients before backward pass
            outputs = model(images)           # Forward pass
            loss = criterion(outputs, labels) # Calculate loss
            loss.backward()                   # Backpropagation
            optimizer.step()                  # Update model parameters
           
            total_loss += loss.item()         # Accumulate loss

        avg_loss = total_loss / len(trainloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Evaluation on test dataset
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradients for evaluation
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Get predicted class index
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # Count correct predictions
   
    accuracy = 100 * correct / total  # Calculate accuracy percentage
    print(f"Test Accuracy: {accuracy:.2f}%")

    # If this model outperforms previous ones, save it and its config
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_config = (hidden_layers, activation_function)

# Display best model configuration and its accuracy
print(f"\nBest Model Configuration: Hidden: {best_config[0]}, Activation: {best_config[1]} with Accuracy: {best_accuracy:.2f}%")

# Interactive prediction: let user pick an image index to test the model on
while True:
    try:
        idx = int(input(f"Enter a test image index (0 to {len(testset)-1}) for prediction: "))
        if 0 <= idx < len(testset):
            break
        else:
            print("Index out of range, try again.")
    except ValueError:
        print("Invalid input, please enter an integer.")

# Retrieve the image and its true label from the test set
img, true_label = testset[idx]

# Convert image tensor to numpy array and unnormalize for display
img_np = img.numpy()
img_np = img_np * 0.5 + 0.5           # Undo normalization
img_np = np.transpose(img_np, (1, 2, 0))  # Change from (C,H,W) to (H,W,C) for matplotlib

# Display the image with its true label as title
plt.imshow(img_np)
plt.title(f"True Label: {classes[true_label]}")
plt.axis('off')
plt.show()

# Make prediction on the selected image using the best trained model
best_model.eval()
with torch.no_grad():
    input_tensor = img.unsqueeze(0).to(device)  # Add batch dimension and move to device
    output = best_model(input_tensor)
    _, predicted_label = torch.max(output, 1)

# Print the predicted label
print(f"Predicted Label by Best Model: {classes[predicted_label.item()]}")
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tabulate import tabulate

# --------------------------
# Configuration parameters
# --------------------------
hidden_layers = (512, 256, 128)   # Sizes of hidden layers in the Feedforward Neural Network
epochs = 7                        # Number of training epochs
batch_size = 128                 # Batch size for training and testing
lr = 0.001                      # Learning rate for the optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else CPU

# --------------------------
# Dataset preparation with normalization
# --------------------------
# Normalize CIFAR-10 images with dataset-specific mean and std to improve training stability
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL images to PyTorch tensors
    transforms.Normalize((0.4914, 0.4822, 0.4465),  # mean for each channel (R,G,B)
                         (0.2023, 0.1994, 0.2010))  # std for each channel (R,G,B)
])

# Download CIFAR-10 training and test datasets with transformations applied
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# DataLoaders for batching and shuffling data during training/testing
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# --------------------------
# Helper function to get activation layers by name
# --------------------------
def get_activation(name):
    # Returns the activation function module by string name
    return {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid()}[name]

# --------------------------
# Feedforward Neural Network Definition
# --------------------------
class FeedforwardNet(nn.Module):
    def __init__(self, hidden_units, activation='relu', dropout=False):
        super().__init__()
        # Define fully connected layers
        self.fc1 = nn.Linear(32*32*3, hidden_units[0])  # Input layer (flattened image size to first hidden layer)
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])  # Second hidden layer
        self.fc3 = nn.Linear(hidden_units[1], hidden_units[2])  # Third hidden layer
        self.fc4 = nn.Linear(hidden_units[2], 10)              # Output layer (10 classes for CIFAR-10)

        self.act = get_activation(activation)  # Choose activation function
        self.do = nn.Dropout(0.5) if dropout else nn.Identity()  # Dropout layer if enabled, else identity (no-op)

    def forward(self, x):
        # Flatten input images from (batch_size, 3, 32, 32) to (batch_size, 3*32*32)
        x = x.view(-1, 32*32*3)

        # Forward pass through layers with activation and optional dropout
        x = self.do(self.act(self.fc1(x)))
        x = self.do(self.act(self.fc2(x)))
        x = self.do(self.act(self.fc3(x)))

        # Final output layer (logits)
        x = self.fc4(x)
        return x

# --------------------------
# Weight Initialization Functions
# --------------------------
def apply_xavier_init(model):
    # Applies Xavier uniform initialization to all Linear layers' weights and zeros biases
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

def apply_kaiming_init(model):
    # Applies Kaiming (He) uniform initialization for ReLU activations to weights and zeros biases
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

# --------------------------
# Training Function
# --------------------------
def train_model(model, optimizer, criterion, epochs):
    model.train()  # Set model to training mode (enables dropout)
    epoch_losses = []
    epoch_accuracies = []

    for epoch in range(epochs):
        running_loss = 0
        correct = 0
        total = 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)  # Move data to GPU if available

            optimizer.zero_grad()           # Clear gradients from previous step
            outputs = model(images)         # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()                 # Backpropagation
            optimizer.step()                # Update weights

            running_loss += loss.item()     # Accumulate batch loss

            # Calculate accuracy for the batch
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(trainloader)  # Average loss over epoch
        accuracy = 100 * correct / total            # Training accuracy in percent

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Train Acc: {accuracy:.2f}%")

        epoch_losses.append(avg_loss)
        epoch_accuracies.append(accuracy)

    return epoch_losses, epoch_accuracies

# --------------------------
# Evaluation Function
# --------------------------
def evaluate_model(model):
    model.eval()  # Set model to evaluation mode (disables dropout)
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient computations
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    return acc

# --------------------------
# Running experiments for different modes
# --------------------------
results = []
best_result = {'method': None, 'accuracy': 0.0}

for mode in ['baseline', 'xavier', 'kaiming', 'dropout', 'l2']:
    print(f"\n--- Running: {mode.upper()} ---")

    # Initialize model with dropout enabled only for 'dropout' mode
    model = FeedforwardNet(hidden_layers, activation='relu', dropout=(mode == 'dropout')).to(device)

    # Apply respective weight initialization if required
    if mode == 'xavier':
        apply_xavier_init(model)
    elif mode == 'kaiming':
        apply_kaiming_init(model)

    # Use weight decay (L2 regularization) only in 'l2' mode
    if mode == 'l2':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification

    # Train the model and get per-epoch losses and accuracies
    epoch_losses, epoch_accuracies = train_model(model, optimizer, criterion, epochs)

    # Evaluate the trained model on test data
    test_acc = evaluate_model(model)

    # Calculate average loss over all epochs
    avg_loss = sum(epoch_losses) / len(epoch_losses)

    # Append results for summary table
    results.append([mode, f"{avg_loss:.4f}", f"{test_acc:.2f}%"])

    # Track best performing mode by test accuracy
    if test_acc > best_result['accuracy']:
        best_result['method'] = mode
        best_result['accuracy'] = test_acc

# --------------------------
# Print summary table and best model info
# --------------------------
print("\n=== Summary Table ===")
headers = ["Mode", "Avg Loss", "Test Accuracy"]
print(tabulate(results, headers=headers, tablefmt="github"))
print(f"\nBest Model: {best_result['method'].upper()} with Accuracy: {best_result['accuracy']:.2f}%")
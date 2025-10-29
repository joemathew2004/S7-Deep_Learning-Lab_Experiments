import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg19
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data preparation
def prepare_data():
    """Prepare MNIST dataset with appropriate transforms for VGG-19"""
    
    # Transform to convert MNIST (1 channel) to RGB (3 channels) and resize to 224x224 for VGG
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                             download=True, transform=transform_train)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, 
                                            download=True, transform=transform_test)
    
    # Use smaller subset for faster training (can be adjusted)
    train_size = 10000
    val_size = 2000
    test_size = 2000
    
    # Split training data
    train_subset, _ = random_split(train_dataset, [train_size, len(train_dataset) - train_size])
    test_subset, _ = random_split(test_dataset, [test_size, len(test_dataset) - test_size])
    val_subset, train_subset = random_split(train_subset, [val_size, train_size - val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader

# Model definitions
class VGG19FixedFeatures(nn.Module):
    """VGG-19 as fixed feature extractor"""
    def __init__(self, num_classes=10):
        super(VGG19FixedFeatures, self).__init__()
        
        # Load pre-trained VGG-19
        self.vgg19 = vgg19(pretrained=True)
        
        # Freeze all parameters
        for param in self.vgg19.parameters():
            param.requires_grad = False
        
        # Replace classifier
        num_features = self.vgg19.classifier[6].in_features
        self.vgg19.classifier[6] = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.vgg19(x)

class VGG19FineTuned(nn.Module):
    """VGG-19 with fine-tuning"""
    def __init__(self, num_classes=10):
        super(VGG19FineTuned, self).__init__()
        
        # Load pre-trained VGG-19
        self.vgg19 = vgg19(pretrained=True)
        
        # Freeze early layers, unfreeze later layers
        # Freeze features up to layer 20 (roughly first 3 blocks)
        for i, param in enumerate(self.vgg19.features.parameters()):
            if i < 20:
                param.requires_grad = False
        
        # Keep classifier trainable
        num_features = self.vgg19.classifier[6].in_features
        self.vgg19.classifier[6] = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.vgg19(x)

class BaselineCNN(nn.Module):
    """Simple CNN baseline for comparison"""
    def __init__(self, num_classes=10):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Calculate the size after convolution and pooling
        # 224 -> 112 -> 56 -> 28
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Training function
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    """Train model and return training history"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    history = defaultdict(list)
    model.to(device)
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.2f}%, '
              f'Val Acc: {val_acc:.2f}%, Time: {epoch_time:.2f}s')
        
        scheduler.step()
    
    return history

# Evaluation function
def evaluate_model(model, test_loader):
    """Evaluate model on test set"""
    model.eval()
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    test_acc = 100. * test_correct / test_total
    return test_acc, all_predictions, all_targets

# Visualization functions
def plot_training_history(histories, model_names):
    """Plot training history comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for history, name in zip(histories, model_names):
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Training & Validation Loss
        axes[0, 0].plot(epochs, history['train_loss'], label=f'{name} - Train')
        axes[0, 1].plot(epochs, history['val_loss'], label=f'{name} - Val')
        
        # Training & Validation Accuracy
        axes[1, 0].plot(epochs, history['train_acc'], label=f'{name} - Train')
        axes[1, 1].plot(epochs, history['val_acc'], label=f'{name} - Val')
    
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    return cm

def compare_models(results):
    """Create comparison visualizations"""
    model_names = list(results.keys())
    test_accuracies = [results[name]['test_acc'] for name in model_names]
    
    # Bar plot of test accuracies
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, test_accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
    plt.title('Model Comparison - Test Accuracy')
    plt.ylabel('Test Accuracy (%)')
    plt.ylim(0, 100)
    
    # Add value labels on bars
    for bar, acc in zip(bars, test_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# Main execution
def main():
    print("=== MNIST Digit Classification with VGG-19 Transfer Learning ===")
    
    # Prepare data
    print("\n1. Preparing data...")
    train_loader, val_loader, test_loader = prepare_data()
    
    # Initialize models
    print("\n2. Initializing models...")
    models = {
        'VGG19-Fixed': VGG19FixedFeatures(num_classes=10),
        'VGG19-FineTuned': VGG19FineTuned(num_classes=10),
        'Baseline-CNN': BaselineCNN(num_classes=10)
    }
    
    # Print model information
    for name, model in models.items():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name}: Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    # Training configuration
    epochs = 8
    results = {}
    histories = []
    model_names = []
    
    # Train each model
    print("\n3. Training models...")
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        
        # Adjust learning rate based on model type
        lr = 0.001 if 'VGG19-Fixed' in name else 0.0001
        
        start_time = time.time()
        history = train_model(model, train_loader, val_loader, epochs=epochs, lr=lr)
        training_time = time.time() - start_time
        
        # Evaluate on test set
        test_acc, predictions, targets = evaluate_model(model, test_loader)
        
        results[name] = {
            'model': model,
            'history': history,
            'test_acc': test_acc,
            'predictions': predictions,
            'targets': targets,
            'training_time': training_time
        }
        
        histories.append(history)
        model_names.append(name)
        
        print(f"{name} - Test Accuracy: {test_acc:.2f}%, Training Time: {training_time:.2f}s")
    
    # Visualizations
    print("\n4. Generating visualizations...")
    
    # Plot training histories
    plot_training_history(histories, model_names)
    
    # Plot confusion matrices
    for name, result in results.items():
        plot_confusion_matrix(result['targets'], result['predictions'], name)
    
    # Model comparison
    compare_models(results)
    
    # Detailed performance analysis
    print("\n=== PERFORMANCE ANALYSIS ===")
    print("-" * 60)
    
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Test Accuracy: {result['test_acc']:.2f}%")
        print(f"  Training Time: {result['training_time']:.2f}s")
        print(f"  Final Training Acc: {result['history']['train_acc'][-1]:.2f}%")
        print(f"  Final Validation Acc: {result['history']['val_acc'][-1]:.2f}%")
        
        # Classification report
        print(f"\nClassification Report for {name}:")
        print(classification_report(result['targets'], result['predictions'], 
                                  target_names=[str(i) for i in range(10)]))
    
    # Analysis insights
    print("\n=== TRANSFER LEARNING ANALYSIS ===")
    print("-" * 60)
    
    fixed_acc = results['VGG19-Fixed']['test_acc']
    finetuned_acc = results['VGG19-FineTuned']['test_acc']
    baseline_acc = results['Baseline-CNN']['test_acc']
    
    print(f"1. Fixed Feature Extractor vs Fine-tuning:")
    print(f"   - VGG19-Fixed: {fixed_acc:.2f}%")
    print(f"   - VGG19-FineTuned: {finetuned_acc:.2f}%")
    print(f"   - Improvement: {finetuned_acc - fixed_acc:.2f}%")
    
    print(f"\n2. Transfer Learning vs Baseline:")
    print(f"   - Best Transfer Learning: {max(fixed_acc, finetuned_acc):.2f}%")
    print(f"   - Baseline CNN: {baseline_acc:.2f}%")
    print(f"   - Transfer Learning Advantage: {max(fixed_acc, finetuned_acc) - baseline_acc:.2f}%")
    
    print(f"\n3. Training Efficiency:")
    for name, result in results.items():
        trainable_params = sum(p.numel() for p in result['model'].parameters() if p.requires_grad)
        print(f"   - {name}: {trainable_params:,} trainable parameters, "
              f"{result['training_time']:.1f}s training time")
    
    print("\n=== KEY INSIGHTS ===")
    print("-" * 60)
    print("• Transfer learning significantly outperforms training from scratch")
    print("• Fine-tuning generally provides better performance than fixed features")
    print("• Pre-trained features capture relevant representations even for different domains")
    print("• Fine-tuning allows adaptation to the specific task while preserving learned features")

if __name__ == "__main__":
    main()

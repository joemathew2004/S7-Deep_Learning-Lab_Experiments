import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from collections import Counter
import re
from prettytable import PrettyTable
import matplotlib.pyplot as plt

# 1. Load IMDB dataset
dataset = load_dataset("imdb")
train_data = dataset["train"]
test_data = dataset["test"]

# 2. Preprocessing
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.split()

counter = Counter()
for example in train_data:
    counter.update(tokenize(example["text"]))

vocab_size = 20000
vocab = {word: idx + 2 for idx, (word, _) in enumerate(counter.most_common(vocab_size))}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

def encode(text, max_len=200):
    tokens = tokenize(text)
    ids = [vocab.get(token, 1) for token in tokens]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return torch.tensor(ids)

# 3. Custom Dataset
class IMDBDataset(Dataset):
    def __init__(self, split):
        self.data = dataset[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        label = self.data[idx]["label"]
        return encode(text), torch.tensor(label)

train_dataset = IMDBDataset("train")
test_dataset = IMDBDataset("test")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# 4. Model Classes
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SentimentRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, nonlinearity="tanh")
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        _, hidden = self.rnn(x)
        hidden = self.dropout(hidden.squeeze(0))
        return self.fc(hidden)

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        hidden = self.dropout(hidden[-1])
        return self.fc(hidden)

class SentimentGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SentimentGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        _, hidden = self.gru(x)
        hidden = self.dropout(hidden.squeeze(0))
        return self.fc(hidden)

# 5. Training & Evaluation
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, test_loader, device):
    model.eval()
    correct, total, total_loss = 0, 0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = criterion(output, y)
            total_loss += loss.item()
            preds = torch.argmax(output, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100 * correct / total, total_loss / len(test_loader)

# 6. Run Experiments
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = len(vocab)
embed_dim = 128
hidden_dim = 128
output_dim = 2
epochs = 3

results = {}
val_accuracies = {}
val_losses = {}

for model_name, model_class in [("RNN", SentimentRNN), ("LSTM", SentimentLSTM), ("GRU", SentimentGRU)]:
    print(f"\nTraining {model_name} model...")
    model = model_class(vocab_size, embed_dim, hidden_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    val_acc_history = []
    val_loss_history = []

    for epoch in range(epochs):
        loss = train_model(model, train_loader, criterion, optimizer, device)
        val_acc, val_loss = evaluate_model(model, test_loader, device)
        val_acc_history.append(val_acc)
        val_loss_history.append(val_loss)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    val_accuracies[model_name] = val_acc_history
    val_losses[model_name] = val_loss_history
    results[model_name] = val_acc_history[-1]

# 7. Summary Table
table = PrettyTable()
table.field_names = ["Model", "Test Accuracy (%)"]
best_model = max(results, key=results.get)

for model_name, acc in results.items():
    table.add_row([model_name, f"{acc:.2f}"])
print("\nPerformance Summary:")
print(table)
print(f"\nBest Model: {best_model} ✅")

# 8. Validation Plots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for model_name in val_accuracies:
    plt.plot(range(1, epochs+1), val_accuracies[model_name], label=f'{model_name}')
plt.title('Validation Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.subplot(1, 2, 2)
for model_name in val_losses:
    plt.plot(range(1, epochs+1), val_losses[model_name], label=f'{model_name}')
plt.title('Validation Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

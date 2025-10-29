import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from collections import Counter
import re

# 1. Load IMDB dataset
dataset = load_dataset("imdb")

train_data = dataset["train"]
test_data = dataset["test"]

# 2. Preprocessing (tokenizer + vocab)
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.split()

counter = Counter()
for example in train_data:
    counter.update(tokenize(example["text"]))

# Limit vocab size
vocab_size = 20000
vocab = {word: idx + 2 for idx, (word, _) in enumerate(counter.most_common(vocab_size))}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

def encode(text, max_len=200):
    tokens = tokenize(text)
    ids = [vocab.get(token, 1) for token in tokens]  # 1 = <UNK>
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))  # pad
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

# 4. Define RNN Model
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SentimentRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, nonlinearity="tanh")
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)                     # [batch, seq, embed_dim]
        _, hidden = self.rnn(x)                   # hidden -> [1, batch, hidden_dim]
        hidden = self.dropout(hidden.squeeze(0))  # [batch, hidden_dim]
        return self.fc(hidden)                    # [batch, output_dim]

model = SentimentRNN(vocab_size=len(vocab), embed_dim=128, hidden_dim=128, output_dim=2)

# 5. Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training started...")
for epoch in range(5):  # keep small for demo
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
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# 6. Evaluation
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        output = model(X)
        preds = torch.argmax(output, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

print(f"Test Accuracy: {100*correct/total:.2f}%")

# 7. User Input Prediction
def predict_review(text):
    model.eval()
    encoded = encode(text).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(encoded)
        pred = torch.argmax(output, dim=1).item()
    return "Positive" if pred == 1 else "Negative"

# Example user test
while True:
    user_inp = input("Enter a movie review (or 'quit' to exit): ")
    if user_inp.lower() == "quit":
        break
    print("Prediction:", predict_review(user_inp))

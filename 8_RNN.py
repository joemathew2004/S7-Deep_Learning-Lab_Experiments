import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# --------------------------
# 1. Load and preprocess data
# --------------------------
tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for label, text in data_iter:
        yield tokenizer(text)

# Load train and test
train_iter = IMDB(split='train')
test_iter = IMDB(split='test')

# Build vocab
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Reload train/test since iterators are exhausted
train_iter = IMDB(split='train')
test_iter = list(IMDB(split='test'))

def collate_batch(batch):
    label_map = {"neg": 0, "pos": 1}
    text_list, label_list = [], []
    for label, text in batch:
        tokens = torch.tensor(vocab(tokenizer(text)), dtype=torch.long)
        text_list.append(tokens)
        label_list.append(label_map[label])
    text_list = pad_sequence(text_list, batch_first=True)
    label_list = torch.tensor(label_list, dtype=torch.long)
    return text_list, label_list

batch_size = 32
train_loader = DataLoader(list(train_iter), batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_iter, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

# --------------------------
# 2. Define RNN Model
# --------------------------
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SentimentRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))

# Model, loss, optimizer
vocab_size = len(vocab)
embed_dim = 64
hidden_dim = 128
output_dim = 2

model = SentimentRNN(vocab_size, embed_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --------------------------
# 3. Training Loop
# --------------------------
for epoch in range(3):
    model.train()
    total_loss, total_correct = 0, 0
    for text, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(text)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += (outputs.argmax(1) == labels).sum().item()
    print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}, Train Acc: {total_correct/len(train_loader.dataset)*100:.2f}%")

# --------------------------
# 4. Evaluation on Test
# --------------------------
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for text, labels in test_loader:
        outputs = model(text)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
print(f"Test Accuracy: {correct/total*100:.2f}%")

# --------------------------
# 5. User input: Predict a review
# --------------------------
def predict_review(index):
    text, label = test_iter[index]
    tokens = torch.tensor(vocab(tokenizer(text)), dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        output = model(tokens)
        prediction = output.argmax(1).item()
    sentiment = "Positive" if prediction == 1 else "Negative"
    print(f"\nReview: {text[:500]}...")  # Show first 500 chars
    print(f"Actual Sentiment: {'Positive' if label == 'pos' else 'Negative'}")
    print(f"Predicted Sentiment: {sentiment}")

# Example usage
user_index = int(input("Enter a review index (0 - 24999): "))
predict_review(user_index)

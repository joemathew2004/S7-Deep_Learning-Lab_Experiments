import os
import random
import unicodedata
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# ---------------------------
# Config / Hyperparameters
# ---------------------------
DATA_PATH = "Dataset_English_Hindi.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MAX_VOCAB = 10000
MAX_LEN_SRC = 20
MAX_LEN_TRG = 20
EMB_SIZE = 128
HIDDEN_SIZE = 256
BATCH_SIZE = 64
epochs = 15
TEACHER_FORCING_RATIO = 0.5
LEARNING_RATE = 0.001
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

# ---------------------------
# Utilities
# ---------------------------
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if unicodedata.category(c) != 'Mn')

def normalize_en(s):
    s = s.lower().strip()
    s = unicode_to_ascii(s)
    # keep punctuation relevant for translation; we will strip other unwanted chars
    s = re.sub(r"[^a-zA-Z0-9\.\!\?\,' ]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_hi(s):
    # minimal normalization for Hindi — keep unicode
    s = s.strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(text):
    # whitespace tokenization (works decently for this small demo)
    return text.split()

# ---------------------------
# Load dataset
# ---------------------------
df = pd.read_csv(DATA_PATH)
# Ensure columns exist
assert 'English' in df.columns and 'Hindi' in df.columns, "CSV must contain 'English' and 'Hindi' columns"

# Preprocess
pairs = []
for _, row in df.iterrows():
    en = str(row['English'])
    hi = str(row['Hindi'])
    en = normalize_en(en)
    hi = normalize_hi(hi)
    if len(en) == 0 or len(hi) == 0:
        continue
    pairs.append((en, hi))

print(f"Loaded {len(pairs)} sentence pairs")

# Shuffle and split
random.shuffle(pairs)
split = int(0.8 * len(pairs))
train_pairs = pairs[:split]
test_pairs = pairs[split:]

# ---------------------------
# Build vocabularies
# ---------------------------
def build_vocab(sentences, max_size):
    counter = Counter()
    for s in sentences:
        counter.update(tokenize(s))
    most_common = counter.most_common(max_size)
    itos = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + [w for w, _ in most_common]
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos

src_sentences = [p[0] for p in train_pairs]
trg_sentences = [p[1] for p in train_pairs]

SRC_stoi, SRC_itos = build_vocab(src_sentences, MAX_VOCAB)
TRG_stoi, TRG_itos = build_vocab(trg_sentences, MAX_VOCAB)

SRC_PAD = SRC_stoi[PAD_TOKEN]
SRC_SOS = SRC_stoi[SOS_TOKEN]
SRC_EOS = SRC_stoi[EOS_TOKEN]
SRC_UNK = SRC_stoi[UNK_TOKEN]

TRG_PAD = TRG_stoi[PAD_TOKEN]
TRG_SOS = TRG_stoi[SOS_TOKEN]
TRG_EOS = TRG_stoi[EOS_TOKEN]
TRG_UNK = TRG_stoi[UNK_TOKEN]

print(f"Source vocab size: {len(SRC_itos)}, Target vocab size: {len(TRG_itos)}")

# ---------------------------
# Encode & Dataset
# ---------------------------
def encode_src(s, max_len=MAX_LEN_SRC):
    toks = tokenize(s)
    ids = [SRC_stoi.get(t, SRC_UNK) for t in toks]
    ids = ids[:max_len-2]  # reserve for sos/eos if you want, but here we won't add sos in encoder
    ids = ids + [SRC_PAD] * (max_len - len(ids))
    return ids

def encode_trg(s, max_len=MAX_LEN_TRG):
    toks = tokenize(s)
    ids = [TRG_SOS] + [TRG_stoi.get(t, TRG_UNK) for t in toks][:max_len-2] + [TRG_EOS]
    ids = ids + [TRG_PAD] * (max_len - len(ids))
    return ids

class TranslationDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        en, hi = self.pairs[idx]
        src_ids = torch.tensor(encode_src(en), dtype=torch.long)
        trg_ids = torch.tensor(encode_trg(hi), dtype=torch.long)
        return src_ids, trg_ids

train_dataset = TranslationDataset(train_pairs)
test_dataset = TranslationDataset(test_pairs)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ---------------------------
# Model: Encoder & Decoder
# ---------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=SRC_PAD)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True)
    def forward(self, src):
        # src: [batch, src_len]
        embedded = self.embedding(src)                # [batch, src_len, emb_dim]
        outputs, hidden = self.gru(embedded)          # outputs [batch, src_len, hid_dim], hidden [1, batch, hid_dim]
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=TRG_PAD)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
    def forward(self, input, hidden):
        # input: [batch] (token ids for current time step)
        input = input.unsqueeze(1)                     # [batch, 1]
        embedded = self.embedding(input)               # [batch,1,emb_dim]
        output, hidden = self.gru(embedded, hidden)    # output [batch,1,hid], hidden [1,batch,hid]
        prediction = self.fc_out(output.squeeze(1))    # [batch, output_dim]
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [batch, src_len], trg: [batch, trg_len]
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = len(TRG_itos)
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        hidden = self.encoder(src)   # [1,batch,hid]
        # first input to decoder: <sos> token for each example
        input = trg[:,0]             # [batch] (should be SOS)
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden)  # output: [batch, vocab]
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
        return outputs

# ---------------------------
# Initialize model, optimizer, loss
# ---------------------------
enc = Encoder(input_dim=len(SRC_itos), emb_dim=EMB_SIZE, hid_dim=HIDDEN_SIZE).to(device)
dec = Decoder(output_dim=len(TRG_itos), emb_dim=EMB_SIZE, hid_dim=HIDDEN_SIZE).to(device)
model = Seq2Seq(enc, dec, device).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD)

# ---------------------------
# Training loop
# ---------------------------
train_losses = []
val_losses = []

def evaluate_bleu(model, loader):
    model.eval()
    refs = []
    preds = []
    with torch.no_grad():
        for src, trg in loader:
            src = src.to(device)
            # greedy decode
            batch_size = src.size(0)
            hidden = model.encoder(src)
            input = torch.tensor([TRG_SOS]*batch_size, dtype=torch.long, device=device)
            outputs = []
            for _ in range(MAX_LEN_TRG):
                out, hidden = model.decoder(input, hidden)
                top1 = out.argmax(1)
                outputs.append(top1.cpu().numpy())
                input = top1
            outputs = np.stack(outputs, axis=1)  # [batch, trg_len]
            for i in range(src.size(0)):
                # reference: trg from loader (has sos/eos/pads) -> convert to tokens w/o sos/eos/pad
                trg_indices = trg[i].cpu().numpy()
                ref_tokens = []
                for idx in trg_indices:
                    tok = TRG_itos[idx] if idx < len(TRG_itos) else UNK_TOKEN
                    if tok == EOS_TOKEN:
                        break
                    if tok not in (SOS_TOKEN, PAD_TOKEN):
                        ref_tokens.append(tok)
                pred_indices = outputs[i]
                pred_tokens = []
                for idx in pred_indices:
                    tok = TRG_itos[idx] if idx < len(TRG_itos) else UNK_TOKEN
                    if tok == EOS_TOKEN:
                        break
                    if tok not in (SOS_TOKEN, PAD_TOKEN):
                        pred_tokens.append(tok)
                refs.append([ref_tokens])
                preds.append(pred_tokens)
    # corpus_bleu expects list of reference lists and list of candidate token lists
    smoothie = SmoothingFunction().method4
    bleu = corpus_bleu(refs, preds, smoothing_function=smoothie)
    return bleu

print("Starting training...")
for epoch in range(1, epochs+1):
    model.train()
    epoch_loss = 0
    for src, trg in train_loader:
        src = src.to(device)
        trg = trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio=TEACHER_FORCING_RATIO)
        # output: [batch, trg_len, vocab]; trg: [batch, trg_len]
        output_dim = output.shape[-1]
        output_flat = output[:,1:,:].reshape(-1, output_dim)   # skip first token (t=0)
        trg_flat = trg[:,1:].reshape(-1)
        loss = criterion(output_flat, trg_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    # compute BLEU on test set every epoch (optional)
    bleu = evaluate_bleu(model, test_loader)
    val_losses.append(0.0)  # placeholder if you want val loss; we compute BLEU instead
    print(f"Epoch {epoch}/{epochs} — Train Loss: {avg_loss:.4f} — Test BLEU: {bleu:.4f}")

# ---------------------------
# Plot loss curve
# ---------------------------
plt.figure(figsize=(8,5))
plt.plot(range(1, len(train_losses)+1), train_losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.show()

# ---------------------------
# Final evaluation (BLEU + print some examples)
# ---------------------------
final_bleu = evaluate_bleu(model, test_loader)
print(f"\nFinal BLEU on test set: {final_bleu:.4f}\n")

# Print a few examples from test set
model.eval()
with torch.no_grad():
    n_examples = 10
    for i in range(n_examples):
        src_text, trg_text = test_pairs[i]
        src_enc = torch.tensor(encode_src(src_text), dtype=torch.long).unsqueeze(0).to(device)
        hidden = model.encoder(src_enc)
        input_tok = torch.tensor([TRG_SOS], dtype=torch.long, device=device)
        decoded_tokens = []
        for _ in range(MAX_LEN_TRG):
            out, hidden = model.decoder(input_tok, hidden)
            top1 = out.argmax(1).item()
            if top1 == TRG_EOS:
                break
            decoded_tokens.append(TRG_itos[top1] if top1 < len(TRG_itos) else UNK_TOKEN)
            input_tok = torch.tensor([top1], dtype=torch.long, device=device)
        print("EN:", src_text)
        print("GT_HI:", trg_text)
        print("PRED_HI:", " ".join(decoded_tokens))
        print("-"*40)

# ---------------------------
# Interactive translation
# ---------------------------
def translate_sentence(sentence, max_len=MAX_LEN_SRC):
    sentence = normalize_en(sentence)
    enc_in = torch.tensor(encode_src(sentence, max_len=MAX_LEN_SRC), dtype=torch.long).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        hidden = model.encoder(enc_in)
        input_tok = torch.tensor([TRG_SOS], dtype=torch.long, device=device)
        decoded = []
        for _ in range(MAX_LEN_TRG):
            out, hidden = model.decoder(input_tok, hidden)
            top1 = out.argmax(1).item()
            if top1 == TRG_EOS:
                break
            decoded.append(TRG_itos[top1] if top1 < len(TRG_itos) else UNK_TOKEN)
            input_tok = torch.tensor([top1], dtype=torch.long, device=device)
    return " ".join(decoded)

print("\nInteractive translation (type 'quit' to exit):")
while True:
    s = input("Enter English sentence: ").strip()
    if s.lower() in ("quit", "exit"):
        break
    print("Predicted Hindi:", translate_sentence(s))

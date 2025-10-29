import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset, DataLoader

# -------------------------------
# 1. Download NIFTY-50 data
# -------------------------------
nifty_data = yf.download('^NSEI', start='2010-01-01', end='2025-01-01')
data = nifty_data[['Close']].values  # Use closing price

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# -------------------------------
# 2. Dataset class
# -------------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, data, lookback=60):
        self.data = data
        self.lookback = lookback

    def __len__(self):
        return len(self.data) - self.lookback

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.lookback]
        y = self.data[idx+self.lookback]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# -------------------------------
# 3. Split train/test
# -------------------------------
lookback = 60
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

train_dataset = TimeSeriesDataset(train_data, lookback)
test_dataset = TimeSeriesDataset(test_data, lookback)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# -------------------------------
# 4. LSTM model
# -------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        pred = self.linear(lstm_out[:, -1, :])  # last hidden state
        return pred

# -------------------------------
# 5. Initialize model, criterion, optimizer
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(input_size=1, hidden_layer_size=50, output_size=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# 6. Training
# -------------------------------
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.view(x_batch.size(0), x_batch.size(1), 1).to(device)  # fix shape
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.6f}")

# -------------------------------
# 7. Evaluation
# -------------------------------
model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.view(x_batch.size(0), x_batch.size(1), 1).to(device)
        y_batch = y_batch.to(device)
        y_pred = model(x_batch)
        predictions.extend(y_pred.cpu().numpy())
        actuals.extend(y_batch.cpu().numpy())

predictions = np.array(predictions).flatten()
actuals = np.array(actuals).flatten()

# Inverse scale to original prices
predictions = scaler.inverse_transform(predictions.reshape(-1,1)).flatten()
actuals = scaler.inverse_transform(actuals.reshape(-1,1)).flatten()

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actuals, predictions))
print(f"\nRoot Mean Squared Error (RMSE): {rmse:.2f}")

print("\nPredictions vs Actuals:")
print("Index\tActual Price\tPredicted Price")
for i in range(7):
    print(f"{i+1}\t{actuals[i]:.2f}\t\t{predictions[i]:.2f}")


# -------------------------------
# 8. Visualization
# -------------------------------
plt.figure(figsize=(12,6))
plt.plot(actuals, label='Actual Prices')
plt.plot(predictions, label='Predicted Prices')
plt.title('NIFTY-50 Price Prediction Using LSTM')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

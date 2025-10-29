import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras import models, layers, optimizers


nifty_data = yf.download('^NSEI', start='2010-01-01', end='2025-01-01')   #  nifty_data = pd.read_csv('nifty_data.csv')  
data = nifty_data[['Close']].values  # Use closing price

# df.columns = df.columns.str.strip().str.lower()

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)


# 2. Prepare sequences
def create_dataset(dataset, lookback=60):           # Convert the time series into samples
    X, y = [], []
    for i in range(len(dataset) - lookback):
        X.append(dataset[i:i+lookback, 0])          # past 60 days of prices
        y.append(dataset[i+lookback, 0])            # next day price
    return np.array(X), np.array(y)

lookback = 60       # 60 days lookback window

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

X_train, y_train = create_dataset(train_data, lookback)
X_test, y_test = create_dataset(test_data, lookback)

# Reshape Data for LSTM [samples, time_steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# Build LSTM model
model = models.Sequential([
    layers.LSTM(50, return_sequences=False, input_shape=(lookback, 1)),
    layers.Dense(1)
])

model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_data=(X_test, y_test),
    verbose=1
)

# 6. Make predictions
predictions = model.predict(X_test)

# Inverse scale
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
print(f"\nRoot Mean Squared Error (RMSE): {rmse:.2f}")


# 7. Plot actual vs predicted prices
plt.figure(figsize=(12,6))
plt.plot(y_test_actual, label='Actual Prices')
plt.plot(predictions, label='Predicted Prices')
plt.title('NIFTY-50 Price Prediction using LSTM (TensorFlow)')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# 8. Plot training & validation loss
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

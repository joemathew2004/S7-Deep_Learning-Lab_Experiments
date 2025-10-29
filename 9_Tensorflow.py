import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, datasets, models

vocab_size = 10000   
max_len = 200     

(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=vocab_size)

x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

def build_model(cell_type="RNN"):
    model = models.Sequential()
    model.add(layers.Embedding(vocab_size, output_dim=128, input_length=max_len))
   
    if cell_type == "RNN":
        model.add(layers.SimpleRNN(units=128, activation='tanh'))
    elif cell_type == "LSTM":
        model.add(layers.LSTM(units=128))
    elif cell_type == "GRU":
        model.add(layers.GRU(units=128))
   
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

models = {"RNN": build_model("RNN"),
          "LSTM": build_model("LSTM"),
          "GRU": build_model("GRU")}

histories = {}
test_results = {}

for name, model in models.items():
    print(f"\nTraining {name} model...")
    history = model.fit(x_train, y_train,
                        epochs=3, 
                        batch_size=64,
                        validation_split=0.2,
                        verbose=1)
   
    histories[name] = history
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    test_results[name] = acc
    print(f"{name} Test Accuracy: {acc:.4f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for name, history in histories.items():
    plt.plot(history.history['val_accuracy'], label=f'{name} Val Acc')
plt.title("Validation Accuracy Comparison")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
for name, history in histories.items():
    plt.plot(history.history['val_loss'], label=f'{name} Val Loss')
plt.title("Validation Loss Comparison")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.bar(test_results.keys(), test_results.values(), color=['blue','green','red'])
plt.title("Test Accuracy of RNN vs LSTM vs GRU")
plt.ylabel("Accuracy")
plt.show()

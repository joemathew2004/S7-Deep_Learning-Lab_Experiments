import tensorflow as tf
from tensorflow.keras import datasets, models, layers
from tensorflow.keras.preprocessing.sequence import pad_sequences


# 1. Load and preprocess data
vocab_size = 10000  # Top 10,000 most frequent words
maxlen = 200        # Cut reviews after 200 words

(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=vocab_size)

# Pad sequences to ensure uniform length
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# 2. Build RNN model
model = models.Sequential([
    layers.Embedding(vocab_size, 128, input_length=maxlen),
    layers.SimpleRNN(128, activation='tanh'),
    layers.Dense(1, activation='sigmoid')
])

# 3. Compile model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 4. Train model
history = model.fit(x_train, y_train,
                    epochs=5,
                    batch_size=64,
                    validation_data=(x_test, y_test))

# 5. Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

# 6. Plot accuracy and loss
import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.show()


# ------------------ Predict User Input ------------------

word_index = tf.keras.datasets.imdb.get_word_index()

# Function to encode user sentence
def encode_review(text):
    words = text.lower().split()
    encoded = [word_index.get(word, 2) for word in words]  # 2 = <UNK> (unknown)
    padded = pad_sequences([encoded], maxlen=maxlen)
    return padded

# Take input from user
user_input = input("\nEnter a movie review: ")
encoded_input = encode_review(user_input)
prediction = model.predict(encoded_input)[0][0]

# Display result
if prediction > 0.5:
    print(f"\nSentiment: 😊 Positive ({prediction:.2f})")
else:
    print(f"\nSentiment: 😞 Negative ({prediction:.2f})")
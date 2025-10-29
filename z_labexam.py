import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import models, layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("spam.csv", encoding='latin-1')        #encoding

# df.columns = ['label', 'message']
   
df = df[['message','label']]

# Encode labels (ham = 0, spam = 1)
encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])

texts = df['message'].values
labels = df['label'].values

# -------------------------------
# 2. Tokenize and pad sequences
# -------------------------------
vocab_size = 10000
maxlen = 100

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')


X_train, X_test, y_train, y_test = train_test_split(padded, labels, test_size=0.2, random_state=42)


model = models.Sequential([
    layers.Embedding(vocab_size, 64, input_length=maxlen),
    layers.GRU(64, return_sequences=False),
    layers.Dropout(0.5),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=1)


loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {acc*100:.2f}%")


while True:
    user_input = input("\nEnter an SMS to classify (or 'exit' to quit): ").strip()
    if user_input.lower() == 'exit':
        break

    seq = tokenizer.texts_to_sequences([user_input])
    padded_input = pad_sequences(seq, maxlen=maxlen, padding='post', truncating='post')
    pred = model.predict(padded_input)[0][0]

    if pred > 0.5:
        print("Prediction: 🚨 Spam Message")
    else:
        print("Prediction: ✅ Ham (Not Spam)")

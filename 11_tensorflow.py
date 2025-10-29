# Shallow English to Hindi Translator Demo using TensorFlow/Keras
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Load and prepare a small subset of the dataset for demo
data = pd.read_csv('Dataset_English_Hindi.csv').dropna().sample(200, random_state=42)
eng_texts = data['English'].astype(str).tolist()
hin_texts = ['<start> ' + t + ' <end>' for t in data['Hindi'].astype(str).tolist()]

# 2. Tokenize and pad
MAX_VOCAB = 2000
eng_tok = Tokenizer(num_words=MAX_VOCAB)
eng_tok.fit_on_texts(eng_texts)
eng_seq = eng_tok.texts_to_sequences(eng_texts)
eng_maxlen = max(len(s) for s in eng_seq)
eng_data = pad_sequences(eng_seq, maxlen=eng_maxlen, padding='post')

hin_tok = Tokenizer(num_words=MAX_VOCAB, filters='')
hin_tok.fit_on_texts(hin_texts)
hin_seq = hin_tok.texts_to_sequences(hin_texts)
hin_maxlen = max(len(s) for s in hin_seq)
hin_data = pad_sequences(hin_seq, maxlen=hin_maxlen, padding='post')


# Prepare decoder input (without last token) and target (without first token)
dec_in_data = pad_sequences([s[:-1] for s in hin_seq], maxlen=hin_maxlen-1, padding='post')
dec_tar_data = pad_sequences([s[1:] for s in hin_seq], maxlen=hin_maxlen-1, padding='post')


# 3. Build shallow encoder-decoder model
latent_dim = 128      # Hidden dimensionality of LSTM
en_inputs = Input(shape=(eng_maxlen,))
en_embed = Embedding(input_dim=MAX_VOCAB, output_dim=latent_dim)(en_inputs)
_, state_h, state_c = LSTM(latent_dim, return_state=True)(en_embed)         # Return only final hidden and cell states
encoder_states = [state_h, state_c]

de_inputs = Input(shape=(hin_maxlen-1,))
de_embed = Embedding(input_dim=MAX_VOCAB, output_dim=latent_dim)(de_inputs)
de_lstm, _, _ = LSTM(latent_dim, return_sequences=True, return_state=True)(de_embed, initial_state=encoder_states)
de_dense = Dense(MAX_VOCAB, activation='softmax')(de_lstm)    # Output layer – predict next word in Hindi sequence

model = Model([en_inputs, de_inputs], de_dense)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 4. Train
model.fit([eng_data, dec_in_data], dec_tar_data, batch_size=32, epochs=10, validation_split=0.2)

# 5. Build inference models
encoder_model = Model(en_inputs, encoder_states)

# Decoder setup for inference
de_state_h = Input(shape=(latent_dim,))
de_state_c = Input(shape=(latent_dim,))
de_inputs2 = Input(shape=(1,))  # One word at a time

# Embedding layer for decoder inference
de_embed2 = Embedding(input_dim=MAX_VOCAB, output_dim=latent_dim)
de_embed_inf = de_embed2(de_inputs2)

# LSTM and Dense layers for decoder
de_lstm2, h2, c2 = LSTM(latent_dim, return_sequences=True, return_state=True)(
    de_embed_inf, initial_state=[de_state_h, de_state_c])
de_dense2 = Dense(MAX_VOCAB, activation='softmax')(de_lstm2)

# Decoder inference model
decoder_model = Model([de_inputs2, de_state_h, de_state_c], [de_dense2, h2, c2])

# Reverse lookup for words
reverse_hin_index = {v: k for k, v in hin_tok.word_index.items()}

def decode_sequence(input_seq):
    # Encode input sequence
    states_value = encoder_model.predict(input_seq)
    # Start token
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = hin_tok.word_index['<start>']
    decoded = []
    for _ in range(hin_maxlen):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_hin_index.get(sampled_token_index, '')
        if sampled_word == '<end>' or not sampled_word:
            break
        decoded.append(sampled_word)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]
    return ' '.join(decoded)

# 6. Demo: show a few translations
print("DEMO TRANSLATIONS:")
for i in range(5):
    input_seq = eng_data[i:i+1]
    translation = decode_sequence(input_seq)
    print(f"English: {eng_texts[i]}")
    print(f"Predicted Hindi: {translation}")
    print(f"Actual Hindi: {hin_texts[i]}")
    print('-'*5)

# 7. Add a function for user input translation
def translate_user_sentence(sentence):
    seq = eng_tok.texts_to_sequences([sentence])
    seq_pad = pad_sequences(seq, maxlen=eng_maxlen, padding='post')
    return decode_sequence(seq_pad)

# 8. Example usage of user input translation
user_input = "How are you?"
print("USER TRANSLATION EXAMPLE:")
print(f"English: {user_input}")
print(f"Predicted Hindi: {translate_user_sentence(user_input)}")

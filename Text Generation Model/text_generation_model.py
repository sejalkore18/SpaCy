import spacy
import keras
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense,LSTM,Embedding

nlp = spacy.load('en',disable=['parser', 'tagger','ner'])

nlp.max_length = 2000000

with open('houn.txt') as file:
    document = file.read()

tokens = [token.text.lower() for token in nlp(document) if token.text not in '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n ']

train_len = 25+1
text_sequences = []

for i in range(train_len, len(tokens)):
    seq = tokens[i-train_len:i]
    text_sequences.append(seq)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequences)
sequences = tokenizer.texts_to_sequences(text_sequences)

vocabulary_size = len(tokenizer.word_counts)

import numpy as np
sequences = np.array(sequences)

def create_model(vocabulary_size, seq_len):
    model = Sequential()
    model.add(Embedding(vocabulary_size, 25, input_length=seq_len))
    model.add(LSTM(150, return_sequences=True))
    model.add(LSTM(150))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(vocabulary_size, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

from keras.utils import to_categorical
X = sequences[:,:-1]
y = sequences[:,-1]
y = to_categorical(y, num_classes=vocabulary_size+1)
model = create_model(vocabulary_size+1, seq_len)

from pickle import dump,load
model.fit(X, y, batch_size=128, epochs=300,verbose=1)
model.save('model.h5')
dump(tokenizer, open('tokenizer', 'wb'))

from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

def generate_text(model, tokenizer, seq_len, seed_text, num_gen_words):

    output_text = []
    input_text = seed_text
    for i in range(num_gen_words):
        encoded_text = tokenizer.texts_to_sequences([input_text])[0]
        pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')
        pred_word_ind = model.predict_classes(pad_encoded, verbose=0)[0]
        pred_word = tokenizer.index_word[pred_word_ind]
        input_text += ' ' + pred_word
        output_text.append(pred_word)
    return ' '.join(output_text)

import random
random.seed(101)
random_pick = random.randint(0,len(text_sequences))
random_seed_text = text_sequences[random_pick]
seed_text = ' '.join(random_seed_text)

generate_text(model,tokenizer,seq_len,seed_text=seed_text,num_gen_words=50)

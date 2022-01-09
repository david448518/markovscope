import numpy as np
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Flatten
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from preproc_rnn import process
import sys
import random
import os

# specify a weights file to just generate without training
weight_file = sys.argv[-1]
# specify sequence length
seq_length = 4

# process the data
X, y, sequences, next_words, words, word_to_idx, idx_to_word = process("data\\horoscopes.csv", seq_length)

# set up model
model = Sequential()
model.add(LSTM(256, input_shape=(
    X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(256))
model.add(Dropout(.2))
# model.add(Flatten())
model.add(Dense(y.shape[1], activation='softmax'))

if weight_file[-4:] == 'hdf5':
    model.load_weights(weight_file)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
else:
    fp = 'checkpoints/weights-{epoch:02d}-{loss:.4f}-bigger.hdf5'
    checkpoint = ModelCheckpoint(
        fp, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(X, y, epochs=30, batch_size=128, callbacks=callbacks_list)


# generate text using sequences from the data as seeds
seed = sequences[np.random.randint(0, len(sequences)-1)]
generated = []
print('seed: ', seed)
for i in range(50):
    x = np.reshape(seed, (1, len(seed), 1))
    x = x / float(len(words))
    pred = model.predict(x, verbose=0)
    idx = np.argmax(pred)
    result = idx_to_word[idx]
    generated.append(result)
    seed.append(idx)
    seed = seed[1:len(seed)]

print(" ".join(generated))

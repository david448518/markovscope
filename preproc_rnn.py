# -*- coding: utf-8 -*-
import pandas as pd
import re
import numpy as np
from keras.utils import np_utils


# get_sequences(["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"], 4)
# ->
# [["The", "quick", "brown", "fox"], ["quick", "brown", "fox", "jumps"], ["brown", "fox", "jumps", "over"], ...]
def get_sequences(s, length):
    for i in range(len(s)-length+1):
        yield [s[i+j] for j in range(length)]


def process(file, seq_length):
    df = pd.read_csv(file, sep='|',
                     names=['ind', 'horo', 'date', 'sign'])
	
	# pattern replacements for weird unicode quirks and escaped characters
    patterns = [("’s", " 's"), ("’ve", " 've"), ("n’t", " n't"), ("’re", " 're"), ("’d", " 'd"), ("’ll", " 'll"),
                (",", " , "), ("!", " ! "), ("\(", " ( "), ("\)", " ) "), ("\?", " ? "), ("\.", " . "), (":", " : "), ('"', ' " ')]
    sequences = []
    dictionary = set([])
    next_words = []
    for idx, row in df.iterrows():
        horo = row['horo']
        if type(horo) == str:

			# clean up horoscope:
            for p in patterns:
                horo = re.sub(p[0], p[1], horo)
			
			# Make everything lowercase and turn it into a list
            horo = horo.lower().split()

			# insert Start token
            horo.insert(0, '_S_')
            ends = []

			# find period characters and store their indices
            for i in range(len(horo)):
                if horo[i] == '.':
                    ends.append(i)
			
			# add end tokes after the periods
            for e in reversed(ends):
                horo.insert(e+1, '_E_')
                horo.insert(e+2, '_S_')
			
			# split the horoscope into sequences
            for seq in get_sequences(horo[:-1], seq_length):
				# append all but the last word of the sequence to sequences
                sequences.append(seq[:-1])
				# append the last word (target value) to next_word
                next_words.append(seq[-1])
                # append word to dictionary
                dictionary = dictionary.union(set(horo))

    print('dictionary size: ', len(dictionary))
    print('amount of sequences: ', len(sequences))


    # translations for words to numbers
    word_to_idx = dict((c, i) for i, c in enumerate(dictionary))
    idx_to_word = dict((i, c) for i, c in enumerate(dictionary))

    # Translate textual words to numbers
    for i in range(len(sequences)):
        for j in range(len(sequences[i])):
            sequences[i][j] = word_to_idx[sequences[i][j]]
        next_words[i] = word_to_idx[next_words[i]]

    # Add extra dimension? Of 1?
    X = np.reshape(sequences, (len(sequences), seq_length-1, 1))
    # normalize? Why?
    X = X / len(dictionary)
    # one-hot encoding
    y = np_utils.to_categorical(next_words)
    
    # X is normalized RNN input with shape(sequence_amt, sequence_length - 1, 1)
    # y is one-hot-encoded target value with shape(sequence_amt, dictionary_size)
    # sequences are the textual representations of X
    # next_words are the textual representations of y
    # word_to_idx is a translation list
    # idx_to_word is a translation list
    return (X, y, sequences, next_words, dictionary, word_to_idx, idx_to_word)

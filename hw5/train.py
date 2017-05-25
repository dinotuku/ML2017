#!/usr/bin/env python
# -- coding: utf-8 --
"""
Multi-class & Multi-label Article Classification
Training Part
"""

import os
import pickle
from argparse import ArgumentParser
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, GRU, Dense, Dropout
from keras.callbacks import ModelCheckpoint, Callback
from keras import optimizers
from keras import backend as K
from utils import read_dataset, dump_history

# np.random.seed(0)

VALIDATION_SPLIT = 0.1
EMBEDDING_DIM = 200
DROPOUT_RATE = 0.5

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
LOG_DIR = os.path.join(BASE_DIR, 'log')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

class History(Callback):
    """ Class for training history """
    def on_train_begin(self, logs=None):
        """ Initialization """
        self.tra_loss = []
        self.val_loss = []
        self.tra_f1 = []
        self.val_f1 = []

    def on_epoch_end(self, epoch, logs=None):
        """ Log training information """
        logs = logs or {}
        self.tra_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.tra_f1.append(logs.get('f1_score'))
        self.val_f1.append(logs.get('val_f1_score'))

def f1_score(y_true, y_pred):
    """ F1 loss function """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    score = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return score

def main():
    """ Main function """
    # Argument Parser
    parser = ArgumentParser()
    parser.add_argument('--bag', action='store_true', help='Use Bag-of-words model')
    args = parser.parse_args()

    train_tags, train_texts = read_dataset('./data/train_data.csv', 'train')
    train_tags = np.array(train_tags)
    test_texts = read_dataset('./data/test_data.csv', 'test')
    texts = np.append(train_texts, test_texts)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)

    tok_file = open('./model/bag_tokenizer.pkl'
                    if args.bag else './model/tokenizer.pkl', 'wb')
    pickle.dump(tokenizer, tok_file)
    tok_file.close()

    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index

    print("Found {:d} unique tokens.".format(len(word_index)))

    if args.bag:
        data = tokenizer.sequences_to_matrix(sequences, mode='freq')
    else:
        data = pad_sequences(sequences)
        # Load glove word embedding
        embeddings_index = {}
        file = open(os.path.join(GLOVE_DIR, "glove.6B.{:d}d.txt".format(EMBEDDING_DIM)))
        for line in file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

        file.close()

        print("Found {:d} word vectors.".format(len(embeddings_index)))

        embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

    max_article_length = data.shape[1]

    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', train_tags.shape)

    # Split the data into a training set and a validation set
    indices = np.arange(train_texts.shape[0])
    np.random.shuffle(indices)
    train_data = data[:train_texts.shape[0]][indices]
    train_tags = train_tags[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * train_texts.shape[0])

    x_train = train_data[:-nb_validation_samples]
    y_train = train_tags[:-nb_validation_samples]
    x_val = train_data[-nb_validation_samples:]
    y_val = train_tags[-nb_validation_samples:]

    # Define model
    rmsprop = optimizers.RMSprop(lr=0.001)
    adam = optimizers.Adam(lr=0.001, decay=1e-6, clipvalue=0.5)
    if args.bag:
        model = Sequential()
        model.add(Dense(256, activation='relu', input_shape=(max_article_length, )))
        model.add(Dropout(DROPOUT_RATE))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(DROPOUT_RATE))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(DROPOUT_RATE))
        model.add(Dense(train_tags.shape[1], activation='sigmoid'))

        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[f1_score])
    else:
        model = Sequential()
        model.add(Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=max_article_length,
                            trainable=False))
        model.add(GRU(256, dropout=DROPOUT_RATE, recurrent_dropout=DROPOUT_RATE,
                      return_sequences=True))
        model.add(GRU(256, dropout=DROPOUT_RATE, recurrent_dropout=DROPOUT_RATE))
        model.add(Dense(256, activation='elu'))
        model.add(Dropout(DROPOUT_RATE))
        model.add(Dense(128, activation='elu'))
        model.add(Dropout(DROPOUT_RATE))
        model.add(Dense(64, activation='elu'))
        model.add(Dropout(DROPOUT_RATE))
        model.add(Dense(train_tags.shape[1], activation='sigmoid'))

        model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=[f1_score])

    model.summary()

    model_name = os.path.join(MODEL_DIR, "{epoch:02d}_bag_model.hdf5"
                              if args.bag else "{epoch:02d}_model.hdf5")
    checkpoint = ModelCheckpoint(model_name, monitor='val_f1_score', verbose=0,
                                 save_best_only=True, mode='max')
    history = History()
    callbacks_list = [checkpoint, history]

    model.fit(x_train, y_train,
              epochs=300,
              batch_size=128,
              validation_data=(x_val, y_val),
              callbacks=callbacks_list)

    dump_history(LOG_DIR, history)
    K.clear_session()

if __name__ == '__main__':

    main()

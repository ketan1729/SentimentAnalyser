from __future__ import print_function

import pandas as pd
from keras.models import Sequential
from keras.layers import Dropout
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Embedding

from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import numpy as np
from numpy.random import RandomState


def train_model(model, x_train, x_val, y_train, y_val):
    cb = [ModelCheckpoint("weights.h5", save_best_only=True, save_weights_only=False)]
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=256, callbacks=cb)
    model.save("model.h5")


def evaluate_model(expected_out, predicted_out):
    expected_categories = [np.argmax(x) for x in expected_out]
    predicted_categories = [np.argmax(x) for x in predicted_out]
    cm = confusion_matrix(expected_categories, predicted_categories)
    print(cm)


def get_model(labels_index, word_index):
    embedded_sequences = get_embedding_layer(word_index)
    model = Sequential([
        embedded_sequences,
        Conv1D(512, 5, activation='relu'),
        AveragePooling1D(5),
        Conv1D(256, 5, activation='relu'),
        AveragePooling1D(5),
        Conv1D(128, 5, activation='relu'),
        MaxPooling1D(5),
        Flatten(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dense(len(labels_index), activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    print(model.summary())
    return model


def get_embedding_layer(word_index):
    embeddings = {}
    glove_fp = "glove/glove.twitter.27B.200d.txt"
    with open(glove_fp, 'r', encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings[word] = coefs

    nb_words = min(20000, len(word_index))
    embedding_matrix = np.zeros((nb_words, 200))

    for word, i in word_index.items():
        if i >= 20000:
            continue
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(nb_words, 200, weights=[embedding_matrix], input_length=1000,
                                trainable=False)
    return embedding_layer


def split_the_data(X_processed, Y_processed):
    indices = np.arange(X_processed.shape[0])
    prng = RandomState(1234567890)
    prng.shuffle(indices)

    X_processed = X_processed[indices]
    Y_processed = Y_processed[indices]

    nb_validation_samples = int(0.2 * X_processed.shape[0])

    x_train = X_processed[:-nb_validation_samples]
    y_train = Y_processed[:-nb_validation_samples]

    x_val = X_processed[-nb_validation_samples:]
    y_val = Y_processed[-nb_validation_samples:]

    return x_train, x_val, y_train, y_val


def tokenize_input(X_raw, Y_raw):
    tnzr = Tokenizer(num_words=20000)
    tnzr.fit_on_texts(X_raw)

    sequences = tnzr.texts_to_sequences(X_raw)

    X_proc = pad_sequences(sequences, maxlen=1000)
    Y_proc = to_categorical(np.asarray(Y_raw), 2)

    return X_proc, Y_proc, tnzr.word_index


def load_data_set():
    cols = ['ItemID', 'Sentiment', 'SentimentSource', 'SentimentText']
    df = pd.read_csv("data/SentimentAnalysisDataset.csv",
                     names=cols,
                     encoding='latin-1',
                     usecols=range(len(cols)),
                     lineterminator="\n")

    from sklearn.utils import shuffle
    df = shuffle(df)

    df['is_positive'] = np.where(df['Sentiment'] == "0", False, True)
    Y = df['is_positive'].tolist()
    X = df['SentimentText'].astype(str).tolist()
    return X, Y


def main():
    labels_index = {'Negative': 0, 'Positive': 1}

    X_raw, Y_raw = load_data_set()
    X_processed, Y_processed, word_index = tokenize_input(X_raw, Y_raw)
    x_train, x_val, y_train, y_val = split_the_data(X_processed, Y_processed)

    model = get_model(labels_index, word_index)
    train_model(model, x_train, x_val, y_train, y_val)

    valid_predicted_out = model.predict(x=x_val, batch_size=256)
    evaluate_model(y_val, valid_predicted_out)


if __name__ == "__main__":
    main()

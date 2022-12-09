from __future__ import print_function
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import matplotlib.pyplot as plt
import numpy as np


def get_predictions():
    data = get_data()
    model = load_model('trained_model/model.h5')
    model.load_weights("trained_model/weights.h5")

    print(model.summary())

    pred_res = []

    for de in data:
        X_raw = de
        # for line in sys.stdin:
        # X_raw.append(line)

        X, word_index = tokenize_data(X_raw)
        mod_predictions = model.predict(x=X, batch_size=128)

        pos = 0
        neg = 0

        for index, txt in enumerate(X_raw):
            is_positive = mod_predictions[index][1] >= 0.65
            if is_positive:
                pos += 1
            else:
                neg += 1
        pred_res.append((pos * 100 / len(de), neg * 100 / len(de)))

    return pred_res


def tokenize_data(X_raw):
    tokenizer = Tokenizer(nb_words=20000)
    tokenizer.fit_on_texts(X_raw)
    sequences = tokenizer.texts_to_sequences(X_raw)
    word_index = tokenizer.word_index
    X_processed = pad_sequences(sequences, maxlen=1000)
    return X_processed, word_index


def get_data():
    data = []
    files = ['test/tweets_bruce_brown.txt',
             'test/tweets_jokic.txt',
             'test/tweets_peoples_choice_awards.txt',
             'test/tweets_ted_cruz.txt',
             'test/tweets_walker_warnock.txt']

    for file in files:
        with open(file, encoding='utf8') as f:
            f_data = []
            text = f.readlines()
            # text = text.split("\n\n")
            for line in text:
                la = line.split("RT")
                for el in la:
                    if len(el) > 2:
                        f_data.append(el)
            if len(f_data) > 0:
                data.append(f_data)
    return data


def plot_graph(gd, plot_name):
    y = np.array(gd)
    mylabels = ["Positive", "Negative"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(y, labels=mylabels, autopct='%.1f%%')
    ax.set_title(plot_name)
    plt.tight_layout()
    plt.savefig("plots2/" + plot_name)


if __name__ == "__main__":
    predictions = get_predictions()
    print(predictions)
    plot_graph(predictions[0], "BruceBrown")
    plot_graph(predictions[1], "NikolaJokic")
    plot_graph(predictions[2], "PeoplesChoiceAwards")
    plot_graph(predictions[3], "TedCruz")
    plot_graph(predictions[4], "WalkerWarnock")

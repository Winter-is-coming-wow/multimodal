# -- coding: utf-8 --
import pandas as pd
import numpy as np

# text preprocessing
from nltk.tokenize import word_tokenize
import re
import pickle

# plots and metrics
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# preparing input to our model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# keras layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, GRU, Dense
def clean_text(data):
    # remove hashtags and @usernames
    data = re.sub(r"(#[\d\w\.]+)", '', data)
    data = re.sub(r"(@[\d\w\.]+)", '', data)

    # tekenization using nltk
    data = word_tokenize(data)

    return data


def loader():
    # Number of labels: joy, anger, fear, sadness, neutral, disgust, surprise
    num_classes = 7

    # Number of dimensions for word embedding
    embed_num_dims = 300

    # Max input length (max number of words)
    max_seq_len = 500

    class_names = ['neutral', 'anger', 'disgust', 'fear', 'joy','sadness','surprise']
    #导入训练集和测试集
    data_train = pd.read_csv('data/data_train.csv', encoding='utf-8')
    data_train=data_train.sample(frac=1).reset_index(drop=True)
    X_train = data_train.text
    y_train = data_train.emotion

    data_test = pd.read_csv('data/data_test.csv', encoding='utf-8')
    data_test=data_test.sample(frac=1).reset_index(drop=True)
    X_test = data_test.text
    y_test = data_test.emotion

    data = data_train.append(data_test, ignore_index=True)

    texts = [' '.join(clean_text(text)) for text in data.text]
    texts_train = [' '.join(clean_text(text)) for text in X_train]
    texts_test = [' '.join(clean_text(text)) for text in X_test]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    with open('cache/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    sequence_train = tokenizer.texts_to_sequences(texts_train)
    sequence_test = tokenizer.texts_to_sequences(texts_test)

    index_of_words = tokenizer.word_index

    # vacab size is number of unique words + reserved 0 index for padding
    vocab_size = len(index_of_words) + 1

    print('Number of unique words: {}'.format(len(index_of_words)))
    print(sequence_train[0])
    print(texts_train[0])
    X_train_pad = pad_sequences(sequence_train, maxlen=max_seq_len)
    X_test_pad = pad_sequences(sequence_test, maxlen=max_seq_len)

    encoding = {
        'neutral': 0,
        'anger': 1,
        'disgust': 2,
        'fear': 3,
        'joy': 4,
        'sadness': 5,
        'surprise': 6
    }

    # Integer labels
    y_train = [encoding[x] for x in data_train.emotion]
    y_test = [encoding[x] for x in data_test.emotion]

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    fname = 'G:/demo/python/practice/multimodal/text/embeddings/wiki-news-300d-1M.vec'
    embedd_matrix = create_embedding_matrix(fname, index_of_words, embed_num_dims)
    # Inspect unseen words
    new_words = 0

    for word in index_of_words:
        entry = embedd_matrix[index_of_words[word]]
        if all(v == 0 for v in entry):
            new_words = new_words + 1

    embedd_layer = Embedding(vocab_size,
                                 embed_num_dims,
                                 input_length=max_seq_len,
                                 weights=[embedd_matrix],
                                 trainable=False)

    gru_output_size = 128
    bidirectional = True

    # Embedding Layer, LSTM or biLSTM, Dense, softmax
    model = Sequential()
    model.add(embedd_layer)

    if bidirectional:
        model.add(Bidirectional(GRU(units=gru_output_size,
                                    dropout=0.2,
                                    recurrent_dropout=0.2)))
    else:
        model.add(GRU(units=gru_output_size,
                      dropout=0.2,
                      recurrent_dropout=0.2))

    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    batch_size = 128
    epochs = 15

    history = model.fit(X_train_pad, y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     validation_data=(X_test_pad, y_test))

    model.save('cache/biLSTM_w2v.h5')

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    with open(filepath,encoding='utf-8',errors='ignore') as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]
    return embedding_matrix


def test(message):
    class_names = ['neutral', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    model=load_model('cache/biLSTM_w2v.h5')
    with open('cache/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    max_seq_len = 500
    seq = tokenizer.texts_to_sequences(message)
    padded = pad_sequences(seq, maxlen=max_seq_len)
    pred=model.predict(padded)
    emotion=np.argmax(pred[0])
    print(class_names[emotion])
test(['I hate you,get out!'])
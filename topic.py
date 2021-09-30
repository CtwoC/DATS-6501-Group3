#%%
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate

import pandas as pd
import numpy as np
import re
import pandas as pd
#%%
casual=pd.read_csv('/Users/chenzichu/Desktop/Capstone/data/Casual_df.csv')
casual.head()

#check if all value in text column are string
all(isinstance(item, str) for item in casual['Normalizaed Review'])

def checker(txt):
    try:
        float(txt)
        return False
    except:
        return True

casual=casual[casual['Normalizaed Review'].apply(checker)]
# %%
review_labels=casual[["Action","Indie","Adventure","RPG","Strategy","Simulation","Casual","Sports","Racing"]]
review_labels.sum(axis=0).plot.bar()
# %%
x=list(casual["Normalizaed Review"])
y=review_labels.values
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
# %%
#convert text inputs into embedded vectors
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1

maxlen = 200

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
# %%
from numpy import array
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()

glove_file = open("/Users/chenzichu/Desktop/Capstone/data/glove.6B.100d.txt", encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
# %%
#model
deep_inputs = Input(shape=(maxlen,))
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(deep_inputs)
LSTM_Layer_1 = LSTM(128)(embedding_layer)
dense_layer_1 = Dense(9, activation='sigmoid')(LSTM_Layer_1)
model = Model(inputs=deep_inputs, outputs=dense_layer_1)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())
# %%
history = model.fit(X_train, y_train, batch_size=128, epochs=5, verbose=1, validation_split=0.2)
# %%
#evaluate on text model
score = model.evaluate(X_test, y_test, verbose=1)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])
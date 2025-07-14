import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam


#df[label] -> gets the column associated with the label. turns it into array.
#df.drop() -> drops the specifier and gets the rest in an array
def read_with_pd(path):
    df = pd.read_csv(path)    
    y = df['label']
    x = df.drop(columns='label').to_numpy() / 255.0

    return x, y


def main():
    x_train, y_train = read_with_pd('./train.csv')

    model = Sequential([
        Dense(units=128, activation='relu'),  #use relu for hidden layers
        Dense(units=64, activation='relu'),
        Dense(units=32, activation='relu'),
        Dense(units=10, activation='softmax')
    ])
    model.compile(
        loss=SparseCategoricalCrossentropy(from_logits=False),  
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    model.fit(x_train, y_train, epochs=10)
    model.save("digit_model.h5")
    



main()
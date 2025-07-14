import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam

import os



#df[label] -> gets the column associated with the label. turns it into array.
#df.drop() -> drops the specifier and gets the rest in an array
def read_with_pd(path):
    df = pd.read_csv(path)    
    y = df['label'].to_numpy()
    x = df.drop(columns='label').to_numpy() / 255.0
    
    # Deterministic splits
    folds = np.array_split(np.arange(42000), 10)

    splits = []
    for i in range(10):
        val_idx = folds[i]
        train_idx = np.concatenate(folds[:i] + folds[i+1:])
        x_train, y_train = x[train_idx], y[train_idx]
        x_val, y_val = x[val_idx], y[val_idx]
        splits.append((x_train, y_train, x_val, y_val))
    return splits



def main(splits, index):

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

    success = 0
    fail = 0
    print(f"TRAINING {index}")
    print('\n')
    x_train, y_train, x_val, y_val = splits[index]
    model.fit(x_train, y_train, epochs=20)
    prediction = np.argmax(model.predict(x_val), axis=1)
    for k in range(len(prediction)):
        if (prediction[k]==y_val[k]):
            success=success+1
        else:
            fail = fail+1
    model.save(f"digit_model{index}.h5")
    with open('success_rates.txt', 'a') as file:
        file.write(f"MODEL {index}'S SUCCES RATE = %{(success*100/(success+fail)):.2f}.\n")
    print('\n')


splits = read_with_pd('./train.csv')
for i in range(10):
    main(splits, i)
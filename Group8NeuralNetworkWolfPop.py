from random import shuffle
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

class_train_path = "D:\\Documents\\School\\CS 4470 -- Artificial Intelligence\\Neural Network Group Project\\CS4470 NN Project Files 2020\\neuralnetworkgroupproject\\*\\*.txt"
files = glob.glob(class_train_path)
class_to_label = {1000:0, 1500:1, 2000:2, 2500:3, 3000:4, 3500:5, 4000:6, 4500:7}
labels = [class_to_label.get(int(cur_file[cur_file.index("series ") + 7:][:4])) for cur_file in files]     # Label will be integer from 0-7

# shuffle data so it doesn't have all data from same class in a row
c = list(zip(files, labels))
shuffle(c)
files, labels = zip(*c)

data_runs = []          # Will be a list of data runs.  Each data run is a list of rows from the data file (each row is a list of the 3 integers from one line of input text e.g. [1959, 760, 20])
for file_name in files:   
    cur_run = []
    skipped_first_line = False
    with open(file_name, 'r') as file:
        for line in file:
            if skipped_first_line:
                cur_row = [int(num) for num in line.split()]
                cur_run.append(cur_row)         # add numbers from line to cur_run list
            else:
                skipped_first_line = True       # First line holds "Year Moose Wolves", skip it
        data_runs.append(cur_run)               # add cur_run to data_runs list

train_data_runs = data_runs[0:int(0.8*len(data_runs))]
train_labels = labels[0:int(0.8*len(labels))]
test_data_runs = data_runs[int(0.8*len(data_runs)):]
test_labels = labels[int(0.8*len(labels)):]

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(60,3)),
    keras.layers.Dense(128, activation="relu"),     # how many units/neurons in the layer? + Activation function
    keras.layers.Dense(8, activation="softmax")    # Num of classifictions? + Activation function
    ])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_data_runs, train_labels, epochs=18)

test_loss, test_acc = model.evaluate(test_data_runs, test_labels)
print("Test accuracy: ", test_acc)
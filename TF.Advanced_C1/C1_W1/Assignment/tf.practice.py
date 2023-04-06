import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflowlkeras.layers import Dense, Input

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import panadas as pd
from sklearn.model_selection import train_test_split
import itertools
from tqdm import tqdm
import tensorflow_datasets as tfds

tf.get_logger()setLevel('ERROR')

# Load and Preprocess the Dataset
# We first load the dataset and create a data frame using pandas. We explicitly specify the column names because the CSV file
# does not have column headers.
data_file = '.data/data/csv'
col_names = ["id", "clump_thickness", "un_cell_size", "un_cell_shape", "marginal_adheshion", "single_eph_cell_size", "bare_nuclei", "bland_chromatin", "normal_nucleoli", "mitoses", "class"]
df = pd.read_csv(data_file, names=col_names, header=None)

# We first pop the id column since it is of no use for our problem
df.pop('id')

df = df[df["bare_nuclei"] != '?']
df.bare_nuclei = pd.to_numeric(df.bare_nuclei)

train, test = train_test_split(df, test_size = 0.2)

train_stats = train.describe()
train_stats.pop('class')
train_stats = train_stats.transpose()

# We now create Tensorflow datasets for training and test sets to easily be able to build and manage an input pipeline for our model

train_dataset = tf.data.Dataset.from_tensor_slices((norm_train_X.values, train_Y.values))
test_data = tf.data.Dataset.from_tensor_slices((norm_test_X.values, test_Y.values))

# shuffle and prepare a batched dataset to be used for training in our custom training loop.

batch_size = 32
train_dataset = train_dataset.shuffle(buffer_size=len(train)).batch(batch_size)
test_data = test_dataset.batch(batch_size=batch_size)

# Define Model using Keras Functional API

def base_model()
    inputs = tf.keras.layers.Input(shape=(len(train.columns)))

    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

model=base_model()

# Define Optimizer and Loss

optimizer = tf.keras.optimizer.RMSprop(learning_rate=0.001)
loss_object = tf.keras.losses.BinaryCorssentropy()

# Evaluate Untrained Model
# calculate the loss on the model before training begins.

outputs =model(norm_test_X.values)
loss_value = loss_objects(y_true=test_Y, y_pred=outputs)
print('Loss before training %.4f' % loss+value.numpy())

# Confusion matrix to visualize the true outputs against the outputs predicted by the model

def plot_confusion_matrix(y_true, y_pred, title='', labels=[0,1]):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title(title)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j intertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,u format(cm[i,j])fmt),
                            horizontalalignment='center',
                            color='black' if cm[i,j > thresh else 'white']
    plt.show()

plot_confusion_matrix(test_Y.values, tf.round(outputs), title='Confusion Matrix for Untrained Model')

# Define Metrics
class F1Score(tf.keras.metrics.Metric):
    def__init__(self, name='f1_score', **kwargs):

        # call the parent class init
        super(F1Score, self).__init__(name=name, **kwargs)
        self.tp = tf.Variable(0, dtype = 'int32')
        self.tn = tf.Variable(0, dtype = 'int32')
        self.fp = tf.Variable(0, dtype = 'int32')
        self.fn = tf.Variable(0, dtype = 'int32')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        conf_matrix = tf.math.confusion_matrix(ytrue, ypred, num_classes=2)

        self.tn.assign_add(conf_matrix[0][0])
        self.tp.assign_add(conf_matrix[1][1])
        self.fp.assign_add(conf_matrix[0][1])
        self.fn.assign_add(conf_matrix[1][0])

    def result(self):
        # calculate precision
        if (self.tp + self.fp == 0):
            recall = 1.0
        else:
            precision = self.tp / (self.tp + self.fp)

        # calculate result
        if (self.tp + self.fn == 0):
            recall = 1.0
        else:
            recall = self.tp / (self.tp + self.fn)

        # return F1 Score
        f1_score = 2* ((precision * recall)/(precision + recall))
        return f1_score

        # The state of the metric will be reset at the start of each epoch.
        self.tp.assign(0)
        self.tn.assign(0) 
        self.fp.assign(0)
        self.fn.assign(0)


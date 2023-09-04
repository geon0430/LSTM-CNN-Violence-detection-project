from data_preprocessing import * 
from model import *
import os
import cv2
import random
import numpy as np
import imgaug.augmenters as iaa
from sklearn.model_selection import train_test_split

# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
plt.style.use("seaborn")
from tensorflow.keras.utils import plot_model

my_LSTM_model = create_model()
plot_model(my_LSTM_model, to_file = 'my_LSTM_model_structure_plot.png', show_shapes = True, show_layer_names = True)

patience = 5
start_lr = 0.00001
min_lr = 0.00001
max_lr = 0.0005
batch_size = 8
rampup_epochs = 5
sustain_epochs = 0
exp_decay = .8

def lrfn(epoch):
    if epoch < rampup_epochs:
        return (max_lr - start_lr)/rampup_epochs * epoch + start_lr
    elif epoch < rampup_epochs + sustain_epochs:
        return max_lr
    else:
        return (max_lr - min_lr) * exp_decay**(epoch-rampup_epochs-sustain_epochs) + min_lr


lr_callback = LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=False)

early_stopping_callback = EarlyStopping(monitor = 'val_accuracy', 
                                        patience = 5, restore_best_weights=True)

checkpoint_filepath = '../models/ModelWeights.h5'

model_checkpoints = ModelCheckpoint(filepath=checkpoint_filepath,
                                        save_weights_only=True,
                                        monitor='val_loss',
                                        mode='min',
                                        verbose = 1,
                                        save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                                  factor=0.8,
                                                  patience=3,
                                                  min_lr=0.00001,
                                                  verbose=1)
callbacks = [lr_callback, early_stopping_callback, reduce_lr]
 
my_LSTM_model_history = my_LSTM_model.fit(x = features_train, y = labels_train, epochs = 20, batch_size = 8 ,
                                             shuffle = True, validation_split = 0.2, callbacks = callbacks)

model_evaluation_history = my_LSTM_model.evaluate(features_test, labels_test)
my_LSTM_model.save("../models/my_LSTM_model.h5")
model_json = my_LSTM_model.to_json()
with open("../modelsmy_LSTM_model.json", "w") as json_file:
    json_file.write(model_json)
my_LSTM_model.save_weights("../models/my_LSTM_model_weights.h5")

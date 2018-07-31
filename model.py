import os
import sys
import cv2
import numpy as np
from random import shuffle
import sklearn
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Conv2D, Activation
from keras.layers.pooling import MaxPooling2D
import csv

data_log = './driving_log.csv'
image_dir = './IMG/'

data = list()
with open(data_log) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        data.append(line)
        
train_data, valid_data = train_test_split(data, test_size=0.2)

def random_augment(image, angle):
    return image, angle

def generate_batch(data, batch_size=32):
    num_samples = len(data)
    while True:
        shuffle(data)
        for offset in range(0, num_samples, batch_size):
            batch = data[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_data in batch:
                center_name = image_dir + batch_data[0].split('/')[-1]
                center_image = cv2.imread(center_name)
                center_angle = float(batch_data[3])
                center_image, center_angle = random_augment(center_image, center_angle)
                
                images.append(center_image)
                angles.append(center_angle)
            X_train = np.array(images)
            Y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, Y_train)
            
train_generator = generate_batch(train_data)
valid_generator = generate_batch(valid_data)

ch, row, col = 3, 160, 320

model = Sequential()
model.add(Lambda(lambda x: x/255, input_shape=(row, col, ch), output_shape=(row, col, ch)))
model.add(Conv2D(32, (3,3), strides=2))
model.add(Activation('relu'))
model.add(Conv2D(64, (3,3), strides=2))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=len(train_data),
                    validation_data=valid_generator, validation_steps=len(valid_data),
                    epochs=3, verbose=1)

model.save('model.h5')
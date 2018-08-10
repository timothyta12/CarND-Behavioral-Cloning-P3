import os
import sys
import csv
import cv2
import numpy as np
from random import shuffle
import sklearn
from sklearn.model_selection import train_test_split
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Activation, BatchNormalization, Cropping2D, Dropout
from keras.layers.pooling import MaxPooling2D
<<<<<<< HEAD
from pathlib import Path, PureWindowsPath, PurePath
=======
>>>>>>> refs/remotes/origin/master

data_log = './data/driving_log.csv'
image_dir = Path('./data//IMG/')

data = list()
with open(data_log) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        data.append(line)
<<<<<<< HEAD

train_data, test_data = train_test_split(data, test_size=0.3)
valid_data, test_data = train_test_split(test_data, test_size=0.5)

def flip(image, angle):
    new_image = cv2.flip(image, 1)
    new_angle = -angle
    return new_image, new_angle

def random_brightness(image, angle):
    multiplier = np.random.uniform(0.5, 1.)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = multiplier*hsv[:,:,2]
    new_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return new_image, angle

def random_augment(image, angle):
    if np.random.randint(0,2) == 1:
        image, angle = flip(image, angle)
    image, angle = random_brightness(image, angle)
=======
        
train_data, valid_data = train_test_split(data, test_size=0.2)

def random_augment(image, angle):
    if np.random.randint(0,2) == 1:
        image = cv2.flip(image, 1)
        angle = -angle
>>>>>>> refs/remotes/origin/master
    return image, angle

def generate_batch(data, batch_size=32):
    num_samples = len(data)
    while True:
        shuffle(data)
        for offset in range(0, num_samples, batch_size):
            batch = data[offset:offset+batch_size]
<<<<<<< HEAD
            steering_offset = 0.13
            images = []
            angles = []
            for center_name, left_name, right_name, steering, throttle, brake, speed in batch:
                center_name = str(image_dir.joinpath(PureWindowsPath(center_name).parts[-1]))
                left_name = str(image_dir.joinpath(PureWindowsPath(left_name).parts[-1]))
                right_name = str(image_dir.joinpath(PureWindowsPath(right_name).parts[-1]))

                steering = float(steering)
                # Center Images
                center_image = cv2.imread(center_name)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_image, center_angle = random_augment(center_image, steering)
                images.append(center_image)
                angles.append(center_angle)
                # Left Images
                left_image = cv2.imread(left_name)
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                left_image, left_angle = random_augment(left_image, steering + steering_offset)
                images.append(left_image)
                angles.append(left_angle)
                # Right Images
                right_image = cv2.imread(right_name)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                right_image, right_angle = random_augment(right_image, steering - steering_offset)
                images.append(right_image)
                angles.append(right_angle)
=======
            
            images = []
            angles = []
            for batch_data in batch:
                center_name = batch_data[0]
                center_image = cv2.imread(center_name)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_data[3])
                center_image, center_angle = random_augment(center_image, center_angle)
                
                images.append(center_image)
                angles.append(center_angle)
>>>>>>> refs/remotes/origin/master
            X_train = np.array(images)
            Y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, Y_train)
            
train_generator = generate_batch(train_data, batch_size=64)
valid_generator = generate_batch(valid_data, batch_size=64)

ch, row, col = 3, 160, 320

model = Sequential()

<<<<<<< HEAD
model.add(Lambda(lambda x: x/255, input_shape=(row, col, ch), output_shape=(row, col, ch)))

model.add(Cropping2D(((20, 20), (0, 0))))
=======
model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(row, col, ch), output_shape=(row, col, ch)))

model.add(Cropping2D(((50, 20), (0, 0))))
>>>>>>> refs/remotes/origin/master

model.add(Lambda(lambda img: K.tf.image.resize_images(img, (64,64))))

model.add(Conv2D(32, (3,3), strides=2, kernel_initializer='truncated_normal', bias_initializer='zeros'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(64, (3,3), strides=2, kernel_initializer='truncated_normal', bias_initializer='zeros'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(128, (3,3), strides=2, kernel_initializer='truncated_normal', bias_initializer='zeros'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(256, (3,3), strides=2, kernel_initializer='truncated_normal', bias_initializer='zeros'))
model.add(BatchNormalization())
model.add(Activation('relu'))          

model.add(Flatten())

model.add(Dense(512, kernel_initializer='truncated_normal', bias_initializer='zeros'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

<<<<<<< HEAD
model.add(Dense(512, kernel_initializer='truncated_normal', bias_initializer='zeros'))
=======
model.add(Dense(256, kernel_initializer='truncated_normal', bias_initializer='zeros'))
>>>>>>> refs/remotes/origin/master
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1, kernel_initializer='truncated_normal', bias_initializer='zeros'))

model.summary()
opt = keras.optimizers.Adam(1e-4)
model.compile(loss='mse', optimizer=opt)
model.fit_generator(train_generator, steps_per_epoch=len(train_data),
                    validation_data=valid_generator, validation_steps=len(valid_data),
                    epochs=1, verbose=1)

model.save('model.h5')
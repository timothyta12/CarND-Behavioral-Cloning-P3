import os
import sys
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Activation, BatchNormalization, Cropping2D, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping
from pathlib import Path, PureWindowsPath, PurePath

data_log = './data/driving_log.csv'
image_dir = Path('./data//IMG/')

data = list()
with open(data_log) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        data.append(line)

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
    return image, angle


def normal_load(data):
    offset = 0.13
    images = []
    angles = []
    for center_name, left_name, right_name, steering, throttle, brake, speed in data:
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
        left_image, left_angle = random_augment(left_image, steering+offset)
        images.append(left_image)
        angles.append(left_angle)
        #Right Images
        right_image = cv2.imread(right_name)
        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
        right_image, right_angle = random_augment(right_image, steering-offset)
        images.append(right_image)
        angles.append(right_angle)

    X_train = np.array(images)
    Y_train = np.array(angles)

    return sklearn.utils.shuffle(X_train, Y_train)


train_features, train_labels = normal_load(train_data)
valid_data = normal_load(valid_data)

ch, row, col = 3, 160, 320

model = Sequential()

model.add(Lambda(lambda x: x/255, input_shape=(row, col, ch), output_shape=(row, col, ch)))

model.add(Cropping2D(((20, 20), (0, 0))))

model.add(Lambda(lambda img: K.tf.image.resize_images(img, (64,64))))

model.add(Conv2D(3, (1,1), kernel_initializer='truncated_normal', bias_initializer='zeros'))
model.add(BatchNormalization())
model.add(Activation('relu'))

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
model.add(Dropout(0.5))

model.add(Dense(128, kernel_initializer='truncated_normal', bias_initializer='zeros'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(128, kernel_initializer='truncated_normal', bias_initializer='zeros'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1, kernel_initializer='truncated_normal', bias_initializer='zeros'))

model.summary()
opt = keras.optimizers.Adam(1e-4)
early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1, min_delta=0.0005)
model.compile(loss='mse', optimizer=opt)
model.fit(train_features, train_labels, validation_data=valid_data,
          callbacks=[early_stop],
          epochs=100, verbose=1, batch_size=32)

test_features, test_labels = normal_load(test_data)
test_loss = model.evaluate(x=test_features, y=test_labels)
print(test_loss)

model.save('model.h5')
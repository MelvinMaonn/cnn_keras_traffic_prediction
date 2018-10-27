from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint, TensorBoard
from keras.applications.resnet50 import ResNet50
from keras import Sequential
from keras.models import Model
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Conv2D, Input, GlobalAveragePooling2D
from metrics import *

import numpy as np
from utils import FLAGS
import read_data as rd

from keras.metrics import mean_absolute_percentage_error

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

batch_size = 60
nb_classes = 1
nb_epoch = 20

# input image dimensions
img_rows, img_cols = 330, 319
# The CIFAR10 images are RGB.
img_channels = 1

num_train_samples = 1021 * 27 - FLAGS.time_step
num_batches_train_per_epoch = int(num_train_samples / FLAGS.batch_size)
num_val_samples = 1021 * 3 - FLAGS.time_step
num_batches_val_per_epoch = int(num_val_samples / FLAGS.batch_size)


lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=5)
csv_logger = CSVLogger('cnn_keras.csv')

# model = ResNet50(include_top=False, weights=None, input_shape=(img_channels, img_rows, img_cols), pooling='max')

# TODO how to change the input channel
inputs = Input(shape=(12, img_rows, img_cols))
x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', data_format='channels_first',  activation="relu")(inputs)
x = MaxPooling2D(pool_size=2, strides=2, padding='same', data_format='channels_first')(x)

x = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', data_format='channels_first',  activation="relu")(inputs)
x = MaxPooling2D(pool_size=2, strides=2, padding='same', data_format='channels_first')(x)

x = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', data_format='channels_first',  activation="relu")(x)
x = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', data_format='channels_first',  activation="relu")(x)
x = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', data_format='channels_first',  activation="relu")(x)
x = MaxPooling2D(pool_size=2, strides=2, padding='same', data_format='channels_first')(x)

x = Conv2D(filters=128, kernel_size=(1,1), data_format='channels_first', activation='relu')(x)
x = Conv2D(filters=128, kernel_size=(3,3), data_format='channels_first', activation='relu')(x)
x = MaxPooling2D(pool_size=2, strides=2, padding='same', data_format='channels_first')(x)

# x = GlobalAveragePooling2D()(x)
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(655, activation="relu")(x)

model = Model(inputs=inputs, outputs=x)




# model.add(Input(shape=(12, img_rows, img_cols)))
# model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', data_format='channels_first'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=2, strides=2, padding='same', data_format='channels_first'))

# model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding='same', data_format='channels_first'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=2, strides=2, padding='same', data_format='channels_first'))
#
# model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same', data_format='channels_first'))
# model.add(Activation('relu'))
#
# model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same', data_format='channels_first'))
# model.add(Activation('relu'))
#
# model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same', data_format='channels_first'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=2, strides=2, padding='same', data_format='channels_first'))
#
# model.add(Flatten())
# model.add(Dense(2048))
# model.add(Activation('relu'))
#
# model.add(Dense(655))
# model.add(Activation('relu'))

model.compile(loss='mae',
              optimizer='adam',
              metrics=[masked_rmse_tf, masked_mae_tf, masked_mape_tf])

ckpt = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=1, save_best_only=True,
                       save_weights_only=False, mode='auto', period=1)
# tensor_board = TensorBoard(log_dir='./logs', histogram_freq=0.1, batch_size=32, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0.1, embeddings_layer_names=None, embeddings_metadata=None)

path_train_image = rd.get_files('/mnt/data5/mm/data/traffic/train_2/')
train_label = np.genfromtxt('/mnt/data5/mm/PycharmProjects/cnn_traffic_prediction/data/800r_train_2.txt')
train_generator = rd.my_generator(path_train_image, train_label, num_train_samples)

path_train_image = rd.get_files('/mnt/data5/mm/data/traffic/test/')
train_label = np.genfromtxt('/mnt/data5/mm/PycharmProjects/cnn_traffic_prediction/data/800r_test.txt')
val_generator = rd.my_generator(path_train_image, train_label, num_val_samples)

model.fit_generator(train_generator,
                    steps_per_epoch=num_batches_train_per_epoch,
                    validation_data=val_generator,
                    validation_steps=num_batches_val_per_epoch,
                    epochs=FLAGS.num_epochs, max_queue_size=100,
                    callbacks=[lr_reducer, early_stopper, csv_logger, ckpt])


'''
train_datagen = ImageDataGenerator(rescale=1. / 255)

val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    '/mnt/data5/mm/data/traffic/train_2/',
    target_size=(img_rows, img_cols),
    # color_mode='grayscale',
    batch_size=FLAGS.batch_size,
    class_mode=None)

validation_generator = val_datagen.flow_from_directory(
    '/mnt/data5/mm/data/traffic/test/',
    target_size=(img_rows, img_cols),
    # color_mode='grayscale',
    batch_size=FLAGS.batch_size,
    class_mode=None)

# Fit the model on the batches generated by datagen.flow().
model.fit_generator(train_generator,
                    steps_per_epoch=num_batches_train_per_epoch,
                    validation_data=validation_generator,
                    # validation_steps=validation_generator.batch_size,
                    epochs=nb_epoch, verbose=1, max_q_size=100,
                    callbacks=[lr_reducer, early_stopper, csv_logger, ckpt])
'''

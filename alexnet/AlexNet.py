# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 09:58:18 2020

@author: amruth
"""



from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import ReduceLROnPlateau
import keras.backend as K


from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, AveragePooling3D, Flatten, Dense, Dropout, Activation, BatchNormalization
from keras import backend as K
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD , Adam
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(validation_split = 0.33)

train_generator = train_datagen.flow_from_directory(
    directory="/home/ubuntu/train_set/",
    target_size=(32, 32),
    color_mode="rgb",
    batch_size=4,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

valid_generator = train_datagen.flow_from_directory(
    directory="/home/ubuntu/train_set/",
    target_size=(32, 32),
    color_mode="rgb",
    batch_size=4,
    class_mode="categorical",
    shuffle=True,
    seed=42
)



l2_reg = 0

# Initialize model
model = Sequential()

# 1st Conv Layer
model.add(Conv2D(96, (11, 11), input_shape=(32,32,3),
    padding='same', kernel_regularizer=l2(l2_reg)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 2nd Conv Layer
model.add(Conv2D(256, (5, 5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 3rd Conv Layer
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 4th Conv Layer
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(1024, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

# 5th Conv Layer
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(1024, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 1st FC Layer
model.add(Flatten())
model.add(Dense(3072))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

# 2nd FC Layer
model.add(Dense(4096))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

# 3rd FC Layer
model.add(Dense(32))
model.add(BatchNormalization())
model.add(Activation('softmax'))


model.compile(loss = 'categorical_crossentropy',
              optimizer = keras.optimizers.Adadelta(),
              metrics = ['accuracy'])

checkpointer_best = ModelCheckpoint("/home/ubuntu/saved-model-{epoch:02d}--cnn.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max',period =3)
csv_logger = CSVLogger('log.csv', append=True, separator=';')


reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 4, verbose = 1, min_delta = 0.0001)


batch_size = 32
epochs = 150

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=150,
                    callbacks=[reduce_lr, csv_logger,checkpointer_best]
)

model.save("model.hdf6")


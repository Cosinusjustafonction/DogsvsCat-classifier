import os
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Flatten , GlobalAveragePooling2D , Conv2D,MaxPool2D,BatchNormalization
from tensorflow.keras.activations import linear, relu, sigmoid
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import glob
import cv2
import numpy as np
import scipy
#data processing
train_ds = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
train_ds = train_ds.flow_from_directory('test_set/test_set', target_size=(64, 64),
                                        batch_size=32, class_mode='binary')

test_ds = ImageDataGenerator(rescale = 1./255)
test_ds = test_ds.flow_from_directory('training_set/training_set', target_size=(64, 64), batch_size=32,
                                      class_mode='binary')
#model creation
tf.random.set_seed(1234) # for consistent results
model = Sequential(
    [
        Flatten(),
        BatchNormalization(),
        Dense(128,activation="relu",name="L1"),
        Dense(64,activation="relu",name="L2"),
        Dense(32,activation="relu",name="L3"),
        Dense(1,activation="sigmoid",name="L4")
    ], name = "my_model"
)





model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x=train_ds, validation_data=test_ds, epochs=40)
test_img = image.load_img('cat-2083492_960_720.jpg', target_size = (64, 64))
img = image.img_to_array(test_img)
img = np.expand_dims(img, axis = 0)
r = model.predict(img)
train_ds.class_indices
if r[0][0] == 1:
    pred = 'dog'
else:
    pred = 'cat'

print(pred)



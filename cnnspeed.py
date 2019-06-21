# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
cnn=Sequential()

def basicnn():
    cnn.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    for i in range(0,4):
        cnn.add(Conv2D(32, (3, 3), activation = 'relu'))
        cnn.add(MaxPooling2D(pool_size = (2, 2)))
        
def widerconv():
    cnn.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Conv2D(64, (3, 3), activation = 'relu'))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Conv2D(128, (3, 3), activation = 'relu'))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Conv2D(32, (3, 3), activation = 'relu'))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    

def bottleneck():
    cnn.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Conv2D(32, (1, 1), activation = 'relu'))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Conv2D(32, (3, 3), activation = 'relu'))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Conv2D(32, (1, 1), activation = 'relu'))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    
def bottlewide():
    cnn.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Conv2D(64, (1, 1), activation = 'relu'))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Conv2D(128, (3, 3), activation = 'relu'))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Conv2D(32, (1, 1), activation = 'relu'))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    
 
cnn.add(Flatten())
cnn.add(Dense(units = 128, activation = 'relu'))
cnn.add(Dense(units = 1, activation = 'sigmoid'))
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])    
i=input("Choose structure: 0)basic cnn 1) widerconv 2)bottleneck 3)bottlewide:")
if i==0:
    basicnn()
elif i==1:
    widerconv()
elif i==2:
    bottleneck()
elif i==3:
    bottlewide()
#fitting data size 
import time 
start=time.clock()
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

cnn.fit_generator(training_set,
                         steps_per_epoch = 500,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)

print(time.clock()-start)
#predictions
import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 0:
    prediction = 'dog'
else:
    prediction = 'cat'



    
    


    
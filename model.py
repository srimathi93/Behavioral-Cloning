import os
import csv
import  cv2
import numpy as np
import tensorflow as tf
import pickle
import keras
import math
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
samples = []

files = ['./driving_log_updated.csv','./driving_log_2.csv', './driving_log_1.csv', './driving_log_4.csv']
#'./driving_log_1.csv', './driving_log_updated.csv',,'./driving_log_2.csv', './driving_log_1.csv'   ,'./driving_log_4.csv', './driving_log_5.csv'

#with open("outfile","wt") as fw:
    #writer = csv.writer(fw)
for file in files:
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None) #skip the headers (row 1)
        for line in reader:
            samples.append(line)
'''with open('./driving_log_updated.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None) #skip the headers (row 1)
    for line in reader:
        samples.append(line)'''

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.3)

import cv2
import numpy as np
import sklearn
import math

def generator(samples, batch_size=64):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            augumented_images, augumented_measurements = [], []
            for batch_sample in batch_samples:
                #name = './IMG/'+batch_sample[0].split('/')[-1]
                name = batch_sample[0]
                #print(name)
                image = cv2.imread(name)
                augumented_images.append(image)
                measurement = float(batch_sample[3])
                #print(measurement)
                augumented_measurements.append(measurement)

                #Left Image path
                left_path = batch_sample[1]
                filename_left = left_path.split('/')[-1]
                left_current_path = left_path
                
                
                #Right Image Path
                right_path = batch_sample[2]
                filename_right = right_path.split('/')[-1]
                right_current_path = right_path
                #print(right_current_path)
                image_left = cv2.imread(left_current_path)
                image_right = cv2.imread(right_current_path)
                #Append images and measurements
                
                
                measurement_left = measurement + 0.28
                measurement_right = measurement - 0.2
                
                image_flipped = np.fliplr(image) # FLip image
                measurement_flipped = -measurement #Flip measurement
                augumented_images.append(image_flipped)
                augumented_measurements.append(measurement_flipped)
                #Append left and right measurements
                augumented_measurements.append(measurement_left)
                augumented_measurements.append(measurement_right)
                augumented_images.append(image_left)
                augumented_images.append(image_right)

            # trim image to only see section with road
            X_train = np.array(augumented_images)
            y_train = np.array(augumented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=64

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

#ch, row, col = 3, 80, 320  # Trimmed image format





from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.layers import activations
from keras.optimizers import Adam
import math




optimizer = keras.optimizers.Adam(lr=1e-5)
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((60,25), (0,0))))
model.add(Convolution2D(24,5,5,subsample = (2,2),activation = "relu"))
#quicker processing
model.add(Convolution2D(36,5,5,subsample = (2,2),activation = "relu"))
model.add(Convolution2D(48,5,5,subsample = (2,2),activation = "relu"))
model.add(Convolution2D(64,3,3,activation = "relu"))
model.add(Convolution2D(64,3,3,activation = "relu"))
model.add(Flatten())
model.add(Dense(180))
model.add(Dropout(0.7))
#model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
#model.add(Flatten())
model.add(Dense(50))
#model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dense(1))



model.compile(loss = 'mse', optimizer = optimizer)

#model.fit(X_train, y_train, validation_split =0.2, shuffle = True, nb_epoch = 3, verbose = 1)
history_object = model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), validation_data=validation_generator,         validation_steps=math.ceil(len(validation_samples)/batch_size),
            epochs=8)
model.save('model.h5')    
#print(history_object.history['loss'])
'''plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')'''
#plt.show()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()
plt.savefig('mean_error.png')


exit()


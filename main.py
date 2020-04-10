# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 13:50:58 2020

@author: animesh-srivastava
"""

#%% Importing libraries

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from time import time
#%% Making the model

classifier = Sequential()

classifier.add(Conv2D(64,(3,3),input_shape = (64,64,3),activation = 'relu'))
classifier.add(MaxPool2D(pool_size=(3,3)))
classifier.add(Conv2D(64,(3,3),activation = 'relu'))
classifier.add(MaxPool2D(pool_size=(3,3)))
classifier.add(Flatten())
classifier.add(Dense(128,activation='relu'))
classifier.add(Dropout(0.3))
classifier.add(Dense(64,activation='relu'))
classifier.add(Dropout(0.3))
classifier.add(Dense(6,activation='softmax'))

classifier.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics=["accuracy"])

classifier.summary()

#%% Image preprocessing

train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('./dataset/seg_train/seg_train',target_size=(64, 64),batch_size=32,class_mode='categorical')
test_set = test_datagen.flow_from_directory('./dataset/seg_test/seg_test',target_size=(64,64),batch_size=32,class_mode='categorical')

#%% Fitting the model

start = time()
history = classifier.fit_generator(training_set,steps_per_epoch=250,epochs=25,validation_data=test_set,validation_steps=800)
end = time()

print(f"Time taken is {(end-start)}s")
#%%
plt.plot(history.history['accuracy'])
#%% Saving the model

classifier.save("model.h5")

#%% Loading from model in case it is saved already

classifier = load_model("model.h5")

#%% Predicting on prediction set 

index = training_set.class_indices
file_names = []
preds = []
for file in os.listdir('./datasetz/seg_pred/seg_pred'):
    test_image = image.load_img('./dataset/seg_pred/seg_pred/'+file,target_size=(64,64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis=0)
    result = classifier.predict(test_image)
    if result[0][0] == 1:
        prediction = 'buildings'
    elif result[0][1] == 1:
        prediction = 'forest'
    elif result[0][2] == 1:
       prediction = 'glacier'
    elif result[0][3] == 1:
        prediction = 'mountain'
    elif result[0][4] == 1:
        prediction = 'sea'
    elif result[0][5] == 1:
        prediction = 'street'
    file_names.append(file)
    preds.append(prediction)
    
predictions = pd.DataFrame(np.row_stack([file_names,preds]).T,columns=["File Name","Prediction"])
predictions.to_csv("predictions.csv")

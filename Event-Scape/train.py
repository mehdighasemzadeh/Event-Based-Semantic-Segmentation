#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 22:09:48 2022

@author: mehdi
"""

          
#=============== set batch size and image size and number of classes=====
batch_size = 8
image_size = (256,512)
number_of_classes  = 8
train_split = 1
test_split  = 1
#======================================================================
import os
def get_dir(path):
    out = []
    arr  = sorted(os.listdir(path))
    for i in arr :
        temp = path + i 
        out.append(temp)
        
    return out


current_dir = os.getcwd()
train_image_path = current_dir + "/Dataset/content/our_dataset/train/image/"   
train_event_path = current_dir + "/Dataset/content/our_dataset/train/event/"
train_label_path = current_dir + "/Dataset/content/our_dataset/train/label/"


test_image_path = current_dir + "/Dataset/content/our_dataset/test/image/"   
test_event_path = current_dir + "/Dataset/content/our_dataset/test/event/"
test_label_path = current_dir + "/Dataset/content/our_dataset/test/label/"




#================ event dir ================
All_Train_Event_dir = get_dir(train_event_path)
All_Test_Event_dir  = get_dir(test_event_path)

#=============== label dir ===================
All_Train_Label_dir = get_dir(train_label_path)
All_Test_Label_dir  = get_dir(test_label_path)       
#============= end create dirctories =======================


#============== split train and test data ==========
import random

random_state = random.randint(0, 50)
'''
print("\n")
print("Random State is : " , random_state)
'''
#====== shuffle images list ==========

#====== shuffle events list ==========
random.seed(random_state)
random.shuffle(All_Train_Event_dir)

random.seed(random_state)
random.shuffle(All_Test_Event_dir)

#====== shuffle mask list ==========
random.seed(random_state)
random.shuffle(All_Train_Label_dir)

random.seed(random_state)
random.shuffle(All_Test_Label_dir)




#================imports ================
#========= imports libs ===================
import numpy as np
import cv2
import keras 
import tensorflow as tf
from matplotlib import pyplot as plt
import glob
import random
import segmentation_models as sm
from keras import backend as K
import tensorflow_addons as tfa

#============ import custom libs ===========
from DataGen import imageLoader
from model import SegModel




#===== in this part we split some traing data  with split size ========

number_of_training_data = int( len(All_Train_Event_dir) * train_split ) 
number_of_testing_data  = int( len(All_Test_Event_dir ) * test_split )  

Train_Image_dir = []
Train_Event_dir = []
Train_Label_dir = []
for i in range(0,number_of_training_data):
    Train_Event_dir.append(All_Train_Event_dir[i])
    Train_Label_dir.append(All_Train_Label_dir[i])
    

Test_Image_dir = []
Test_Event_dir = []
Test_Label_dir = []
for i in range(0,number_of_testing_data):
    Test_Event_dir.append(All_Test_Event_dir[i])
    Test_Label_dir.append(All_Test_Label_dir[i])

#============== load train genrator=========================
aug_train = 1
aug_test  = 0
train_img_datagen = imageLoader(Train_Image_dir,Train_Event_dir , 
                                Train_Label_dir, batch_size,image_size,number_of_classes,aug_train)
#============== load test genrator=========================
val_img_datagen = imageLoader(Test_Image_dir,Test_Event_dir , 
                                Test_Label_dir, batch_size,image_size,number_of_classes,aug_test)



#============== calculate steps =================================
steps_per_epoch     = int (number_of_training_data/batch_size) + 1
val_steps_per_epoch = int (number_of_testing_data/batch_size)  + 1


########## load model from model.py and compile model ================


DiceLoss = sm.losses.DiceLoss()
CategoricalCELoss = sm.losses.CategoricalCELoss()
IoULloss = sm.losses.JaccardLoss()

total_loss =  CategoricalCELoss + DiceLoss
metrics = ['accuracy', sm.metrics.IOUScore(per_image = True)]

model = SegModel(8)

#=================== set optimizer ==============
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=5e-4,
    decay_steps = 3*steps_per_epoch  ,
    decay_rate=0.95)

opt = keras.optimizers.Adam(learning_rate=lr_schedule)


#============== compile model ===============================
model.compile(optimizer = opt , loss = total_loss , metrics = metrics)



#============== save model info =====================
training_path = "training1/cp.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    training_path,
    monitor =  'val_loss',
    verbose = 1,
    save_weights_only =True ,
    mode = 'auto',
    save_freq='epoch',

)





#========== train model ==================
#model.load_weights(training_path)
history=model.fit(train_img_datagen,
          steps_per_epoch=steps_per_epoch,
          epochs=30,
          verbose=1,
          callbacks = [cp_callback] ,
          validation_data=val_img_datagen,
          validation_steps=val_steps_per_epoch,
          )



#========== train model ==================
#============== compile model ===============================
model.compile(optimizer = opt , loss = DiceLoss , metrics = metrics)
model.load_weights(training_path)
history=model.fit(train_img_datagen,
          steps_per_epoch=steps_per_epoch,
          epochs=30,
          verbose=1,
          callbacks = [cp_callback] ,
          validation_data=val_img_datagen,
          validation_steps=val_steps_per_epoch,
          )








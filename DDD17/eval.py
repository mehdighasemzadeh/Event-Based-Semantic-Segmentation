#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 13:52:08 2022

@author: mehdi
"""
import os
current_dir = os.getcwd()

event_test_path = current_dir + "/Dataset/dataset_our_codification/events/test/"
image_test_path = current_dir + "/Dataset/dataset_our_codification/images/test/"
label_test_path = current_dir + "/Dataset/dataset_our_codification/labels/test/"
save_path       = current_dir + "/output/"       
    
#=============== set batch size and image size and number of classes=====
batch_size = 1
image_size = (192,320)
number_of_classes  = 6
train_split = 1
test_split  = 1
#======================================================================


def Make_All_Path(path):
    output = []
    arr  = sorted(os.listdir(path))
    for i in arr : 
        dir_e = path + i
        output.append(dir_e)
    return output
    

#================imports ================
#========= imports libs ===================
import os
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
from matplotlib import pyplot as plt
#============ import custom libs ===========
from DataGen import imageLoader
from model import SegModel


loss1 = sm.losses.DiceLoss()
loss2 = sm.losses.CategoricalCELoss()
total_loss = loss1  + loss2

metrics = ['accuracy', sm.metrics.IOUScore(per_image = True)]
model = SegModel(6)



#=================== set optimizer ==============
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=5e-4,
    decay_steps = 550 * 3 ,
    decay_rate=0.9)

opt = keras.optimizers.Adam(learning_rate=lr_schedule)


#============== compile model ===============================
model.compile(optimizer = opt , loss = total_loss , metrics = metrics)
#============== load model  =====================
load_weight_path = "training1/cp.ckpt"
model.load_weights(load_weight_path)


def EventToImage(input_event , output) :
    temp = input_event[:,:,0:1]
    min_temp = np.amin(temp)
    temp = temp - min_temp 
    max_temp = np.amax(temp)
    temp = temp / max_temp
    temp = temp*255
    temp1 = temp.astype(np.uint8)
    output[0:192,0:320,0:1] = temp1
    output[0:192,0:320,1:2] = temp1
    output[0:192,0:320,3:2] = temp1
    return output


   
def ImageToImage(input_image , output) :
    temp = input_image.reshape(192,320,1)
    output[0:192 , 320:640 , 0:1] = temp
    output[0:192 , 320:640 , 1:2] = temp
    output[0:192 , 320:640 , 2:3] = temp
    return output

   
def PrdictToImage(label , output):
    # ===== label = 0 ============
    temp = np.zeros((192,320,3) , np.uint8)
    for i in range(0,192):
        for j in range(0,320):
            
            if label[i][j] == 0 :
                temp[i:i+1 , j : j +1 , :]  = [128,64,128]
            
            if label[i][j] == 1 :
                temp[i:i+1 , j : j +1 , :]  = [70,70,70] 

            if label[i][j] == 2 :
                temp[i:i+1 , j : j +1 , :]  = [0,255,255]

            if label[i][j] == 3 :
                temp[i:i+1 , j : j +1 , :]  = [35,142,107]
                
            if label[i][j] == 4 :
                temp[i:i+1 , j : j +1 , :]  = [60,20,220]
                
            if label[i][j] == 5 :
                temp[i:i+1 , j : j +1 , :]  = [142,0,0]

    output[0:192,640:960,:] = temp
    
    return output
                
 


def GTToImage(label , output):
    # ===== label = 0 ============
    temp = np.zeros((192,320,3) , np.uint8)
    for i in range(0,192):
        for j in range(0,320):
            
            if label[i][j] == 0 :
                temp[i:i+1 , j : j +1 , :]  = [128,64,128]
            
            if label[i][j] == 1 :
                temp[i:i+1 , j : j +1 , :]  = [70,70,70] 

            if label[i][j] == 2 :
                temp[i:i+1 , j : j +1 , :]  = [0,255,255]

            if label[i][j] == 3 :
                temp[i:i+1 , j : j +1 , :]  = [35,142,107]
                
            if label[i][j] == 4 :
                temp[i:i+1 , j : j +1 , :]  = [60,20,220]
                
            if label[i][j] == 5 :
                temp[i:i+1 , j : j +1 , :]  = [142,0,0]

    output[0:192,960:,:] = temp
    
    return output
                
       

#================= prediction of special sequence =========
#=========== set start and end in below for prdiction ====           
#============ predict =======================================
name = "rec1487417411_export_" 
start = 3600
end   = 4430
for i in range(start,end):
    image_output = np.zeros((192,1280,3) , np.uint8)
    input_event = np.load(event_test_path + name + str(i) + ".npy")
    #============== resizing part ===============
    x2 = cv2.resize(input_event,  (320,192) , interpolation = cv2.INTER_LINEAR)
    image_output = EventToImage(x2 , image_output)
    X2 = x2.reshape(1,192,320,6)
    x1 = cv2.resize(x2,  (160,96) , interpolation = cv2.INTER_LINEAR)
    X1 = x1.reshape(1,96,160,6)
    
    test = model.predict([X1,X2])
    out = np.argmax(test, axis=3)
    out1 = out.reshape(192,320)
    #========== prdicte to image ======================
    image_output = PrdictToImage(out1 , image_output)
    
    #================ image to image ================
    img  = cv2.imread(image_test_path + name + str(i) + ".png" , 0 )
    img1 = cv2.resize(img,  (320,192) , interpolation = cv2.INTER_LINEAR)
    image_output = ImageToImage(img1,image_output)
    
    #========== GT to image ================
    img  = cv2.imread(label_test_path + name + str(i) + ".png"  ,0)
    img1 = cv2.resize(img,  (320,192) , interpolation = cv2.INTER_NEAREST)
    image_output = GTToImage(img1,image_output)
    
    #========= write info on images ===================
    
    image_output = cv2.line(image_output, (639,0), (639,192), (255,255,255), 3)
    image_output = cv2.line(image_output, (319,0), (319,192), (255,255,255), 3)
    image_output = cv2.line(image_output, (959,0), (959,192), (255,255,255), 3)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    image_output = cv2.putText(image_output, 'Event', (10,15), font, 
                   0.5, (255,255,255), 1, cv2.LINE_AA)
    
    image_output = cv2.putText(image_output, 'Image', (335,15), font, 
                   0.5, (255,255,255), 1, cv2.LINE_AA)
    
    image_output = cv2.putText(image_output, 'P', (645,15), font, 
                   0.5, (255,255,255), 1, cv2.LINE_AA)
    
    image_output = cv2.putText(image_output, 'GT', (965,15), font, 
                   0.5, (255,255,255), 1, cv2.LINE_AA)
    
    
    cv2.imwrite(save_path + name + str(i) + ".png", image_output)
    #plt.imshow(image_output, interpolation='nearest')
    #plt.show()
    





#============ save video ========================
width  = 1280
height = 192
#this fourcc best compatible for avi
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
save_video = current_dir +'/test.avi' 
video=cv2.VideoWriter(save_video , fourcc, 20.0, (width,height))

Image_path = save_path 
def get_dir(path):
    out = []
    arr  = sorted(os.listdir(path))
    for i in arr :
        temp = path + i 
        out.append(temp)
        
    return out

output_Image_dir = get_dir(Image_path)


for i in range(0,len(output_Image_dir)):
     x=cv2.imread(output_Image_dir[i])
     video.write(x)

cv2.destroyAllWindows()
video.release()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 22:20:23 2022

@author: mehdi
"""

# ========== In this version of DataGen we use shuffle for overfitting ==========
#============== imports ============
import numpy as np
import random
import cv2 


def Label_gen(img):
    out = np.zeros((256,512,8),np.uint8)
    
    out[:,:,0] = np.logical_or(img==0, img==3)

    out[:,:,1] = np.logical_or(img==1 , img==11 , img==2)

    out[:,:,2] = (img==4)
    
    out[:,:,3] = np.logical_or(img==6 , img==7)


    out[:,:,4] = (img==10)
    
    out[:,:,5] = np.logical_or(img==5,img==12 )

    
    out[:,:,6] = (img==9)
    
    out[:,:,7] = (img==8)
    
    return out
  





'''
##################################################################
#=================== start data genrator =========================
#==== size : (256,512) ===========
#----------------- function for loading image from dataset ---------
def load_img(img_dir , resize , aug , flip , x_c , y_c , random_aug):
    images=[]
    for i in range(0,len(img_dir)):
        if aug == 0:
            #===== load image ====
            image = cv2.imread(img_dir[i],cv2.IMREAD_COLOR)
            #====== resize image ==
            #image_resizes = cv2.resize(image,  (256,128) , interpolation = cv2.INTER_LINEAR)
            #==== normalize =======
            out = np.zeros((256,512,3) , np.float32)
            for j in range(0,3):
                temp = image[:,:,j:j+1]
                max_i = np.amax(temp)
                min_i = np.amin(temp)
                temp1 = (temp - min_i)
                if max_i - min_i > 0 :
                    temp1 = temp1/(max_i - min_i)
                temp2 = temp1.astype(np.float32)
                out[:,:,j:j+1] = temp2
            
            #====== append to batch ===========
            images.append(out)
        else : 
            if random_aug[i] == 0 : 
                image = cv2.imread(img_dir[i],cv2.IMREAD_COLOR)
                #====== resize image ==
                #temp1 = cv2.resize(image,  (320,160) , interpolation = cv2.INTER_LINEAR)
                
                temp1 = image
                #=========== flip randomly ==============
                if flip[i] == True:
                    temp1 = np.fliplr(temp1)

                image_resizes1 = temp1[y_c[i]:y_c[i] + 220 , x_c[i] : x_c[i]  + 440, : ]
                image_resizes  = cv2.resize(image_resizes1,  (512,256) , interpolation = cv2.INTER_LINEAR)
                out = np.zeros((256,512,3) , np.float32)
                for j in range(0,3):
                    temp = image_resizes[:,:,j:j+1]
                    max_i = np.amax(temp)
                    min_i = np.amin(temp)
                    temp1 = (temp - min_i)
                    if max_i - min_i > 0 :
                        temp1 = temp1/(max_i - min_i)
                    temp2 = temp1.astype(np.float32)
                    out[:,:,j:j+1] = temp2
                
                images.append(out)
                
                
            if random_aug[i] == 1 : 
                image = cv2.imread(img_dir[i],cv2.IMREAD_COLOR)
                #====== resize image ==
                #temp1 = cv2.resize(image,  (320,160) , interpolation = cv2.INTER_LINEAR)
                temp1 = image
                #=========== flip randomly ==============
                if flip[i] == True:
                    temp1 = np.fliplr(temp1)

                temp2 = temp1[y_c[i]: , x_c[i] : , : ]
                image_resizes = cv2.resize(temp2,  (512,256) , interpolation = cv2.INTER_LINEAR)
                out = np.zeros((256,512,3) , np.float32)
                for j in range(0,3):
                    temp = image_resizes[:,:,j:j+1]
                    max_i = np.amax(temp)
                    min_i = np.amin(temp)
                    temp1 = (temp - min_i)
                    if max_i - min_i > 0 :
                        temp1 = temp1/(max_i - min_i)
                    temp2 = temp1.astype(np.float32)
                    out[:,:,j:j+1] = temp2
                    
                images.append(out)

    images = np.array(images , np.float32)
    return images


'''







#----------------- function for loading image from dataset ---------
def load_event(img_dir , resize , aug , flip , x_c , y_c , random_aug):
    #==== size must be (160,320) ======
    images1  = []
    images2  = []
    #======== no aug ==================
    if aug == 0 :
        for i in range(0,len(img_dir)):    
            #===== load numpy array =======
            image = np.load(img_dir[i])
            #============== resizing part ===============
            #out1 = cv2.resize(image,  (256,128) , interpolation = cv2.INTER_LINEAR)
            out1 = image
            #============ append to batch ======
                
            images1.append(out1)
            
            out1_2 = cv2.resize(out1,  (256,128) , interpolation = cv2.INTER_LINEAR)
            
            images2.append(out1_2)
            
        images1_1 = np.array(images1 , np.float32) 
        images1_2 = np.array(images2 , np.float32) 
        return images1_1 , images1_2
    
    else :
        for i in range(0,len(img_dir)):    
            if random_aug[i] == 0:
                #===== load numpy array =======
                image = np.load(img_dir[i])
                #============== resizing part ===============
                #temp1 = cv2.resize(image,  (320,160) , interpolation = cv2.INTER_LINEAR)
                temp1 = image
                #=========== flip randomly ==============
                if flip[i] == True:
                    temp1 = np.fliplr(temp1)

                out1 = temp1[y_c[i]:y_c[i] + 220 , x_c[i] : x_c[i]  + 440, : ]
                out2 = cv2.resize(out1,  (512,256) , interpolation = cv2.INTER_LINEAR)

                #============ append to batch ======
                images1.append(out2)
                
                out1_2 = cv2.resize(out2,  (256,128) , interpolation = cv2.INTER_LINEAR)
                images2.append(out1_2)

        
        
            if random_aug[i] == 1:
                #===== load numpy array =======
                image = np.load(img_dir[i])
                #============== resizing part ===============
                #temp1 = cv2.resize(image,  (320,160) , interpolation = cv2.INTER_LINEAR)
                temp1 = image
                #=========== flip randomly ==============
                if flip[i] == True:
                    temp1 = np.fliplr(temp1)

                
                temp2 = temp1[ y_c[i] : , x_c[i] : , : ]

                out1  = cv2.resize(temp2,  (512,256) , interpolation = cv2.INTER_LINEAR)


                #============ append to batch ======
                images1.append(out1)
                
                out1_2 = cv2.resize(out1,  (256,128) , interpolation = cv2.INTER_LINEAR)
                images2.append(out1_2)
                
                

                
            
        images1_1 = np.array(images1 , np.float32) 
        images1_2 = np.array(images2 , np.float32) 
        return images1_1 , images1_2
        
    
        
        
            




#----------------- function for loading label from dataset ---------
def load_msk(img_dir , resize , aug , flip , x_c , y_c , random_aug):
    images=[]
    if aug == 0 :
        for i in img_dir:   
            #===== load numpy array =======
            #======== make label ==========
            img = cv2.imread(i,0)
            image = Label_gen(img)
            #======== end make label ======
            #image = np.load(i)
            #========== resize ============
            #out = cv2.resize(image, (256,128), interpolation = cv2.INTER_NEAREST)
            #===== change mask type
            out = image
            out1 = out.astype(np.float32)
            #===== append to batch ============
            images.append(out1)
        
        images1 = np.array(images , dtype = np.float32)
        return images1
    
    else :
        for i in range(0, len(img_dir)):
            if random_aug[i] == 0 :
                #===== load numpy array =======
                #============ make label gen =======
                img = cv2.imread(img_dir[i],0)
                image = Label_gen(img)
                #========== end label gen ==========
                #image = np.load(img_dir[i])
                
                #========== resize ============
                #temp1 = cv2.resize(image, (320,160), interpolation = cv2.INTER_NEAREST)
                temp1 = image
                #=========== flip randomly ==============
                if flip[i] == True :
                    temp1 = np.fliplr(temp1)
                
                #==== simple crop =================
                out = temp1[ y_c[i] : y_c[i] + 220 , x_c[i] : x_c[i] + 440, : ]
                out2 = cv2.resize(out, (512,256), interpolation = cv2.INTER_NEAREST)

                #===== change mask type
                out1 = out2.astype(np.float32)
                #===== append to batch ============
                images.append(out1)
            
            if random_aug[i] == 1 :
                #===== load numpy array =======
                #============ make label gen =======
                img = cv2.imread(img_dir[i],0)
                image = Label_gen(img)
                #========== end label gen ==========
                #image = np.load(img_dir[i])
                #========== resize ============
                #temp1 = cv2.resize(image, (320,160), interpolation = cv2.INTER_NEAREST)
                temp1 = image
                #=========== flip randomly ==============
                if flip[i] == True:
                    temp1 = np.fliplr(temp1)
                
                #==== simple crop =================
                temp2 = temp1[ y_c[i] : , x_c[i]:, : ]
                #========== resize ============
                out = cv2.resize(temp2, (512,256), interpolation = cv2.INTER_NEAREST)

                #===== change mask type
                out1 = out.astype(np.float32)
                #===== append to batch ============
                images.append(out1)
            
        images1 = np.array(images , dtype =  np.float32)
        return images1
            




#----------------- data genrator function ---------
def imageLoader(train_image_dir,train_event_dir , Train_mask_dir, batch_size,size,NumberOfClass,aug):
    

    L = len(train_event_dir)

    #keras needs the generator infinite, so we will use while true  
    while True:
        
        #================== shuffle data in each epoch ===============
        #============= set a random number for random state ============= 
        random_state = random.randint(0, 50)
        #============ shuffle image data ================================
        random.seed(random_state)
        random.shuffle(train_image_dir)
        #============ shuffle event data ================================
        random.seed(random_state)
        random.shuffle(train_event_dir)
        #============ shuffle event data ================================    
        random.seed(random_state)
        random.shuffle(Train_mask_dir)
        """
        print("\n")
        print("random state is : " , random_state)
        """

        #=========== this is genrator part ============
        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            
            #============ make aug data ==========
            x_c = []
            y_c = []
            flip= [] 
            random_aug = []
            
            batch_len = limit - batch_start 
            
            for i in range(0,batch_len):
                temp_random = random.randint(0, 72)
                x_c.append(temp_random)
                temp_random = random.randint(0, 36)
                y_c.append(temp_random)
                temp_random =  bool(random.getrandbits(1))
                flip.append(temp_random)
                temp_random = random.randint(0, 1)
                random_aug.append(temp_random)
                
                
            E1 , E2 = load_event(train_event_dir[batch_start:limit],size,aug,flip , x_c , y_c , random_aug)
            Y  = load_msk(Train_mask_dir[batch_start:limit],size,aug,flip , x_c , y_c , random_aug)

            yield [E2 , E1] , Y # a list and label passed to model #  

            batch_start += batch_size   
            batch_end += batch_size
#=========================== end data genrator part ==========================


#======================== test =========================
'''
Path = "/home/mehdi/Downloads/test/"
label_path = "/home/mehdi/Downloads/test/label/"
event_path = "/home/mehdi/Downloads/test/event/"
image_path = "/home/mehdi/Downloads/sequence_0/rgb/data/"

import os
image_dir = []
event_dir = []
label_dir = []

arr  = sorted(os.listdir(label_path))
for i in arr:
    label_dir.append(label_path+i)

arr  = sorted(os.listdir(event_path))
for i in arr:
    event_dir.append(event_path+i)

arr  = sorted(os.listdir(image_path))
for i in range(0,4):
    image_dir.append(image_path + arr[i])


x , y = next( imageLoader(image_dir , event_dir,label_dir, 5,1,8,1))

I = x[0]
E = x[1]

I0 = I[0,:,:,:] * 255
I1 = I0.astype(np.uint8)

E0 = E[0,:,:,:]

L0 = y[0,:,:,:]

from matplotlib import pyplot as plt
plt.imshow(I1, interpolation='nearest')
plt.show()

for i in range(0,8):
    plt.imshow(E0[:,:,i:i+1], interpolation='nearest')
    plt.show()
    

for i in range(0,8):
    plt.imshow(L0[:,:,i:i+1], interpolation='nearest')
    plt.show()
    


'''


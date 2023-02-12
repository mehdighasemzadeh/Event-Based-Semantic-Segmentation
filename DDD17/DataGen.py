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






##################################################################
#=================== start data genrator =========================
#==== size : (256,128) ===========
#----------------- function for loading image from dataset ---------
def load_img(img_dir,size):
    images=[]
    for i in img_dir:    
        #===== load image ====
        image = cv2.imread(i,cv2.IMREAD_COLOR)
        #====== resize image ==
        image_resizes = cv2.resize(image,  (256,128) , interpolation = cv2.INTER_LINEAR)
        #==== normalize =======
        out = np.zeros((128,256,3) , np.float32)
        for j in range(0,3):
            temp = image_resizes[:,:,j:j+1]
            max_i = np.amax(temp)
            min_i = np.amin(temp)
            temp1 = (temp - min_i)
            if max_i - min_i > 0 :
                temp1 = temp1/(max_i - min_i)
            temp2 = temp1.astype(np.float32)
            out[:,:,j:j+1] = temp2
        
        #====== append to batch ===========
        images.append(out)
    images = np.array(images)
    return(images)










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
            out1 = cv2.resize(image,  (320,192) , interpolation = cv2.INTER_LINEAR)
            out2 = cv2.resize(out1,  (160,96) , interpolation = cv2.INTER_LINEAR)
            #============ append to batch ======
            images1.append(out1)
            images2.append(out2)
    
            
        images3 = np.array(images1 , np.float32)
        images4 = np.array(images2 , np.float32)
        return images3 , images4
    
    else :
        for i in range(0,len(img_dir)):    
            if random_aug[i] == 0:
                #===== load numpy array =======
                image = np.load(img_dir[i])
                #============== resizing part ===============
                temp1 = image

                #=========== flip randomly ==============
                if flip[i] == True:
                    temp1 = np.fliplr(temp1)

                out1_temp = temp1[y_c[i]:y_c[i] + 180 , x_c[i] : x_c[i]  + 306, : ]
                out1 = cv2.resize(out1_temp,  (320,192) , interpolation = cv2.INTER_LINEAR)               
                out2 = cv2.resize(out1,  (160,96) , interpolation = cv2.INTER_LINEAR)

                #============ append to batch ======
                images1.append(out1)
                images2.append(out2)
        
        
            if random_aug[i] == 1:
                #===== load numpy array =======
                image = np.load(img_dir[i])
                #============== resizing part ===============
                temp1 = image
                #=========== flip randomly ==============
                if flip[i] == True:
                    temp1 = np.fliplr(temp1)

                
                temp2 = temp1[ y_c[i] : , x_c[i] : , : ]

                out1  = cv2.resize(temp2,  (320,192) , interpolation = cv2.INTER_LINEAR)
                out2  = cv2.resize(out1 ,  (160,96) , interpolation = cv2.INTER_LINEAR)

                #============ append to batch ======
                images1.append(out1)
                images2.append(out2)
        
                
        images3 = np.array(images1 , dtype = np.float32)
        images4 = np.array(images2 , dtype = np.float32)
            
        return images3 , images4
        
        
        
            




#----------------- function for loading label from dataset ---------
def load_msk(img_dir , resize , aug , flip , x_c , y_c , random_aug):
    images=[]
    if aug == 0 :
        for i in img_dir:   
            #===== load image mask =======
            temp1 = cv2.imread(i,0)
            image = np.zeros((200,346,6) , np.float32)
            for c in range(6):
                image[:,:,c] = (temp1==c)
            
            #========== resize ============
            out = cv2.resize(image, (320,192), interpolation = cv2.INTER_NEAREST)
            #===== change mask type
            out1 = out.astype(np.float32)
            #===== append to batch ============
            images.append(out1)
        
        images1 = np.array(images , dtype = np.float32)
        return images1
    
    else :
        for i in range(0, len(img_dir)):
            if random_aug[i] == 0 :
                
                #===== load image mask =======
                temp1 = cv2.imread(img_dir[i],0)
                image = np.zeros((200,346,6) , np.float32)
                for c in range(6):
                    image[:,:,c] = (temp1==c)
                
                #========== resize ============
                temp1 = image
                #=========== flip randomly ==============
                if flip[i] == True :
                    temp1 = np.fliplr(temp1)
                
                #==== simple crop =================
                out_temp = temp1[ y_c[i] : y_c[i] + 180 , x_c[i] : x_c[i] + 306, : ]
                out = cv2.resize(out_temp, (320,192), interpolation = cv2.INTER_NEAREST)
                #===== change mask type
                out1 = out.astype(np.float32)
                #===== append to batch ============
                images.append(out1)
            
            if random_aug[i] == 1 :
                
                #===== load image mask =======
                temp1 = cv2.imread(img_dir[i],0)
                image = np.zeros((200,346,6) , np.float32)
                for c in range(6):
                    image[:,:,c] = (temp1==c)
                
                
                #========== resize ============
                temp1 = image
                #=========== flip randomly ==============
                if flip[i] == True:
                    temp1 = np.fliplr(temp1)
                
                #==== simple crop =================
                temp2 = temp1[ y_c[i] : , x_c[i]:, : ]
                #========== resize ============
                out = cv2.resize(temp2, (320,192), interpolation = cv2.INTER_NEAREST)

                #===== change mask type
                out1 = out.astype(np.float32)
                #===== append to batch ============
                images.append(out1)
            
        images1 = np.array(images , dtype =  np.float32)
        return images1
            




#----------------- data genrator function ---------
def imageLoader(train_event_dir , Train_mask_dir, batch_size,size,NumberOfClass,aug):
    

    L = len(train_event_dir)

    #keras needs the generator infinite, so we will use while true  
    while True:
        
        #================== shuffle data in each epoch ===============
        #============= set a random number for random state ============= 
        random_state = random.randint(0, 50)
        #============ shuffle event data ================================
        random.seed(random_state)
        random.shuffle(train_event_dir)
        #============ shuffle event data ================================    
        random.seed(random_state)
        random.shuffle(Train_mask_dir)
        #print("\n")
        #print("random state is : " , random_state)


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
                temp_random = random.randint(0, 40)
                x_c.append(temp_random)
                temp_random = random.randint(0, 20)
                y_c.append(temp_random)
                temp_random =  bool(random.getrandbits(1))
                flip.append(temp_random)
                temp_random = random.randint(0, 1)
                random_aug.append(temp_random)
                
                
               
            E1 , E2 = load_event(train_event_dir[batch_start:limit],size,aug,flip , x_c , y_c , random_aug)
            Y = load_msk(Train_mask_dir[batch_start:limit],size,aug,flip , x_c , y_c , random_aug)

            yield [E2,E1] , Y # a list and label passed to model #  

            batch_start += batch_size   
            batch_end += batch_size
#=========================== end data genrator part ==========================



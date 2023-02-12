
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

#============ import custom libs ===========
from model import SegModel


def get_dir(path):
    out = []
    arr  = sorted(os.listdir(path))
    for i in arr :
        temp = path + i 
        out.append(temp)
        
    return out


current_dir = os.getcwd()
Image_path = current_dir + "/Dataset/content/our_dataset/test/image/"   
Event_path = current_dir + "/Dataset/content/our_dataset/test/event/"
Label_path = current_dir + "/Dataset/content/our_dataset/test/label/"



loss1 = sm.losses.DiceLoss()
loss2 = sm.losses.CategoricalCELoss()
total_loss = loss1  + loss2

metrics = ['accuracy', sm.metrics.IOUScore(per_image = True)]

model = SegModel(8)



#=================== set optimizer ==============
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=5e-4,
    decay_steps = 2200 * 3 ,
    decay_rate=0.95)

opt = keras.optimizers.Adam(learning_rate=lr_schedule)


#============== compile model ===============================
model.compile(optimizer = opt , loss = total_loss , metrics = metrics)

#========== train model ==================
load_weight_path = current_dir + "/training1/cp.ckpt"
model.load_weights(load_weight_path)



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



image_ext = "image.png"
event_ext = "events.npy"
label_ext = "gt_labelIds.png"


def EventToImage(input_event , output) :
    temp = input_event[:,:,0:1]
    temp = temp*255
    temp1 = temp.astype(np.uint8)
    output[0:256,0:512,0:1] = temp1
    output[0:256,0:512,1:2] = temp1
    output[0:256,0:512,3:2] = temp1
    return output



def ImageToImage(input_image , output) :
    output[0:256 , 512:1024 , :] = input_image
    return output

   
def PrdictToImage(label , output):
    # ===== label = 0 ============
    temp = np.zeros((256,512,3) , np.uint8)
    for i in range(0,256):
        for j in range(0,512):
            if label[i][j] == 1 :
                temp[i:i+1 , j : j +1 , :]  = [70,70,70] 

            if label[i][j] == 2 :
                temp[i:i+1 , j : j +1 , :]  = [60,20,220]
                
            if label[i][j] == 3 :
                temp[i:i+1 , j : j +1 , :]  = [128,64,128]

            if label[i][j] == 4 :
                temp[i:i+1 , j : j +1 , :]  = [142,0,0]

            if label[i][j] == 5 :
                temp[i:i+1 , j : j +1 , :]  = [153,153,153]

            if label[i][j] == 6 :
                temp[i:i+1 , j : j +1 , :]  = [35,142,107]
                
            if label[i][j] == 7 :
                temp[i:i+1 , j : j +1 , :]  = [232,35,244] 
                
    output[256:512,0:512,:] = temp
    
    return output
                
 


def GTToImage(label , output):
    # ===== label = 0 ============
    temp = np.zeros((256,512,3) , np.uint8)
    for i in range(0,256):
        for j in range(0,512):
            if label[i][j] == 1 :
                temp[i:i+1 , j : j +1 , :]  = [70,70,70] 

            if label[i][j] == 2 :
                temp[i:i+1 , j : j +1 , :]  = [60,20,220]
                
            if label[i][j] == 3 :
                temp[i:i+1 , j : j +1 , :]  = [128,64,128]

            if label[i][j] == 4 :
                temp[i:i+1 , j : j +1 , :]  = [142,0,0]

            if label[i][j] == 5 :
                temp[i:i+1 , j : j +1 , :]  = [153,153,153]

            if label[i][j] == 6 :
                temp[i:i+1 , j : j +1 , :]  = [35,142,107]
                
            if label[i][j] == 7 :
                temp[i:i+1 , j : j +1 , :]  = [232,35,244] 
                
          
    output[256:512,512:,:] = temp
    
    return output

                
          
'''
########################################################
#============= for Single sample =======================   
test_out = np.zeros((512,1024,3) , np.uint8)
name = "05_168_0155_"
start = 0
end   = 1
for i in range(start,end):
    #======= read event ==========================
    input_event  = np.load(Event_path+name+event_ext)
    input_event1 =  cv2.resize(input_event,  (256,128) , interpolation = cv2.INTER_LINEAR)
    #=========== put input image ==================
    test_out = EventToImage(input_event , test_out)
    
    #============ prediction part ===============
    event  = np.reshape(input_event  , (1,256,512,8))
    event1 = np.reshape(input_event1 , (1,128,256,8))
    #============== read image and scale it ===============
    input_image = cv2.imread(Image_path+name+image_ext,cv2.IMREAD_COLOR)
    test_out = ImageToImage(input_image , test_out) 


    test = model.predict([event1,event])
    

    out = np.argmax(test, axis=3)
    out1 = out.reshape(256,512)
    test_out = PrdictToImage(out1,test_out)
    
    l = cv2.imread(Label_path+name+label_ext, 0)
    l1 = Label_gen(l)
    l2 = np.argmax(l1, axis=2)
    l3 = l2.reshape(256,512)
    
    test_out = GTToImage(l3,test_out)


cv2.imwrite("/home/mehdi/Desktop/e3.png", test_out)

plt.imshow(test_out, interpolation='nearest')
plt.show()  
'''










#============= video creator for each sequence =================
#============ set start and end in below for prediction ========


def num_gen(start,end,name):
    out = list()
    for i in range(start , end):
        if i%7 == 1 :
            i1   = str(i).zfill(4)
            temp = name + i1 + "_"
            out.append(temp)
    return out

########################################################
#============= for video sample =======================  

save_path = current_dir + "/output/" 
name_s = "05_298_"
start = 0
end   = 247
name_list = num_gen(start,end,name_s)

for name in name_list:
    test_out = np.zeros((512,1024,3) , np.uint8)
    #======= read event ==========================
    input_event  = np.load(Event_path+name+event_ext)
    input_event1 =  cv2.resize(input_event,  (256,128) , interpolation = cv2.INTER_LINEAR)
    #=========== put input image ==================
    test_out = EventToImage(input_event , test_out)
    
    #============ prediction part ===============
    event  = np.reshape(input_event  , (1,256,512,8))
    event1 = np.reshape(input_event1 , (1,128,256,8))
    #============== read image and scale it ===============
    input_image = cv2.imread(Image_path+name+image_ext,cv2.IMREAD_COLOR)
    test_out = ImageToImage(input_image , test_out) 


    test = model.predict([event1,event])
    

    out = np.argmax(test, axis=3)
    out1 = out.reshape(256,512)
    test_out = PrdictToImage(out1,test_out)
    
    l = cv2.imread(Label_path+name+label_ext, 0)
    l1 = Label_gen(l)
    l2 = np.argmax(l1, axis=2)
    l3 = l2.reshape(256,512)
    
    test_out = GTToImage(l3,test_out)

    
    #========= write info on images ===================
    image_output = cv2.line(test_out, (511,0), (511,511), (255,255,255), 3)
    image_output = cv2.line(image_output, (0,254), (1024,254), (255,255,255), 3)


    font = cv2.FONT_HERSHEY_SIMPLEX
    image_output = cv2.putText(image_output, 'Event', (10,15), font, 
                   0.5, (255,255,255), 1, cv2.LINE_AA)
    
    image_output = cv2.putText(image_output, 'Image', (517,15), font, 
                   0.5, (255,255,255), 1, cv2.LINE_AA)
    
        
    image_output = cv2.putText(image_output, 'P', (10,270), font, 
                   0.5, (255,255,255), 1, cv2.LINE_AA)
    
    image_output = cv2.putText(image_output, 'GT', (517,270), font, 
                   0.5, (255,255,255), 1, cv2.LINE_AA)



    cv2.imwrite(save_path + name + image_ext, image_output)

    #plt.imshow(image_output, interpolation='nearest')
    #plt.show()  





#============== save video ============================
#============= create videos ================================================
width  = 1024
height = 512
#this fourcc best compatible for avi
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
video=cv2.VideoWriter( current_dir + "/test.avi" ,fourcc, 10.0, (width,height))

Image_path = current_dir + "/output/"
output_Image_dir = get_dir(Image_path)


for i in range(0,len(output_Image_dir)):
     x=cv2.imread(output_Image_dir[i])
     video.write(x)

cv2.destroyAllWindows()
video.release()








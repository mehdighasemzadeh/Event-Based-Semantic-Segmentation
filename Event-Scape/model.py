#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 22:19:41 2022

@author: mehdi
"""


#================== imports =========================================
import tensorflow as tf
import numpy as np
from keras import Model
from keras.layers import Input
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv2D , concatenate , Conv2DTranspose ,  MaxPool2D, Flatten, GlobalAveragePooling2D,  BatchNormalization, Layer, Add
from keras.models import Sequential






#################### define ResNet Blosk ######################################
class ResnetBlock(Model):
    
    def __init__(self, channels: int, down_sample=False):

        super().__init__()

        self.__channels = channels
        self.__down_sample = down_sample
        self.__strides = [2, 1] if down_sample else [1, 1]

        KERNEL_SIZE = (3, 3)
        # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
        INIT_SCHEME = "he_normal"

        self.conv_1 = Conv2D(self.__channels, strides=self.__strides[0],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
       
        self.bn_1 = BatchNormalization()
       
        self.conv_2 = Conv2D(self.__channels, strides=self.__strides[1],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
       
        self.bn_2 = BatchNormalization()
        
        self.merge = Add()

        if self.__down_sample:
            # perform down sampling using stride of 2, according to [1].
            self.res_conv = Conv2D(
                self.__channels, strides=2, kernel_size=(1, 1), kernel_initializer=INIT_SCHEME, padding="same")
            self.res_bn = BatchNormalization()

    def call(self, inputs):
        res = inputs

        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x)

        if self.__down_sample:
            res = self.res_conv(res)
            res = self.res_bn(res)

        # if not perform down sample, then add a shortcut directly
        x = self.merge([x, res])
        out = tf.nn.relu(x)
        return x , out
    
#################### end ##############################################




  
############## this class is used for frist layer in encoder for passing to ResNet Blocks ######### 
class layer1(Model):
    
    def __init__(self, channels):
        super().__init__()
        
        self.channels = channels
        
        KERNEL_SIZE = (5, 5)
        # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
        INIT_SCHEME = "he_normal"
        
        self.conv_1 = Conv2D(self.channels, strides=1,
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
       
        self.bn_1 = BatchNormalization()
        

    def call(self, inputs):
        
        x = self.conv_1(inputs)
        x = self.bn_1(x)
        out = tf.nn.relu(x)
        
        return x , out
   
################################## end ####################################################
  

  

############## this class is used for 1x1 conv layers ####################### 
class conv1x1(Model):
    
    def __init__(self, channels):
        super().__init__()
        
        self.channels = channels
        
        KERNEL_SIZE = (1, 1)
        # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
        INIT_SCHEME = "he_normal"
        
        self.conv_1 = Conv2D(self.channels, strides=1,
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
       
        self.bn_1 = BatchNormalization()
        

    def call(self, inputs):
        
        x = self.conv_1(inputs)
        x = self.bn_1(x)
        out = tf.nn.relu(x)
        
        return out
    
################################## end ####################################################












############### this block is used for conv in upsample part or in decoder ##################
class upsample_conv(Model):
    
    def __init__(self, channels):
        super().__init__()
        
        self.channels = channels
        
        KERNEL_SIZE = (3, 3)
        # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
        INIT_SCHEME = "he_normal"
        
        self.conv_1 = Conv2D(self.channels, strides=1,
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
       
        self.bn_1 = BatchNormalization()
        
        self.conv_2 = Conv2D(self.channels, strides=1,
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
       
        self.bn_2 = BatchNormalization()
        

    def call(self, inputs):
        
        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = tf.nn.relu(x)
        
        x = self.conv_2(x)
        x = self.bn_2(x)
        out = tf.nn.relu(x)
        
        return out
    

################################ end #################################################




##=================== main segmantation model ========================## 

class SegModel(Model):
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        #============ set filter size ==============
        filter_size = [64,64,128,256,512]
        
        #==== E : event layer , F : frame ======#
        
        '''
        #========== define max_ploing ==============
        self.maxpolingE = tf.keras.layers.MaxPool2D( pool_size=(3, 3), strides=(2, 2), padding='same')
        self.maxpolingI = tf.keras.layers.MaxPool2D( pool_size=(3, 3), strides=(2, 2), padding='same')
        '''
        
        #============ event encoder layer ==================#

        self.layer1E = layer1(filter_size[0])
        
        self.res1_1E = ResnetBlock(filter_size[1] ,down_sample=True)
        self.res1_2E = ResnetBlock(filter_size[1])
        
        self.res2_1E = ResnetBlock(filter_size[2], down_sample=True)
        self.res2_2E = ResnetBlock(filter_size[2])
        
        
        self.res3_1E = ResnetBlock(filter_size[3], down_sample=True)
        self.res3_2E = ResnetBlock(filter_size[3])
        
        self.res4_1E = ResnetBlock(filter_size[4], down_sample=True)
        self.res4_2E = ResnetBlock(filter_size[4])
        

        #============ Frame encoder layer ==================#
        self.layer1F = layer1(filter_size[0])
        
        self.res1_1F = ResnetBlock(filter_size[1] , down_sample=True)
        self.res1_2F = ResnetBlock(filter_size[1])
        
        self.res2_1F = ResnetBlock(filter_size[2], down_sample=True)
        self.res2_2F = ResnetBlock(filter_size[2])
        
        
        self.res3_1F = ResnetBlock(filter_size[3], down_sample=True)
        self.res3_2F = ResnetBlock(filter_size[3])
        
        self.res4_1F = ResnetBlock(filter_size[4], down_sample=True)
        self.res4_2F = ResnetBlock(filter_size[4])
        
        
        
        #========== 1x1 conv layers for comb Event and Frame with batch norm ======
        self.conv1x1_0 = conv1x1(filter_size[0])
        self.conv1x1_1 = conv1x1(filter_size[1])
        self.conv1x1_2 = conv1x1(filter_size[2])
        self.conv1x1_3 = conv1x1(filter_size[3])
        self.conv1x1_4 = conv1x1(filter_size[4])
        self.conv1x1_5 = conv1x1(filter_size[4])
        
        

        

        #=============== conv in decoder ========================================
        self.convU00 = upsample_conv(int(filter_size[0]/2)) 
        self.convU0 = upsample_conv(filter_size[0]) 
        self.convU1 = upsample_conv(filter_size[1]) 
        self.convU2 = upsample_conv(filter_size[1])
        self.convU3 = upsample_conv(filter_size[2])
        self.convU4 = upsample_conv(filter_size[3])

        #============== deconv layer ==========================================
        
        self.deconv1 = tf.keras.layers.UpSampling2D(size=(2, 2),  interpolation="bilinear")
        self.deconv2 = tf.keras.layers.UpSampling2D(size=(2, 2),  interpolation="bilinear")
        self.deconv3 = tf.keras.layers.UpSampling2D(size=(2, 2),  interpolation="bilinear")
        self.deconv4 = tf.keras.layers.UpSampling2D(size=(2, 2),  interpolation="bilinear")
        self.deconv5 = tf.keras.layers.UpSampling2D(size=(2, 2),  interpolation="bilinear")


        #================= final layer =========================================
        self.FinalLayer = Conv2D(self.channels, (1, 1), activation='softmax')



    def call(self, inputs):
        
        #========= load evnet data and frame data ===========
        I = inputs[0]
        E = inputs[1]
        
        #==== insted of Frame , we pass Event data with scale 1/2 =====
        #================ frist conv layer ============
        xe , xe_r = self.layer1E(E)             # 1
        xf , xf_r = self.layer1F(I)       
        
        #======== frist ResNet Block ==================
        xe1_1 , xe1_1_r = self.res1_1E(xe_r)     # 1/2
        xe1_2 , xe1_2_r = self.res1_2E(xe1_1_r)
 
        xf1_1 , xf1_1_r = self.res1_1F(xf_r)
        xf1_2 , xf1_2_r = self.res1_2F(xf1_1_r)
        
        #======== second ResNet Block ==================
        xe2_1 , xe2_1_r = self.res2_1E(xe1_2_r)  #1/4
        xe2_2 , xe2_2_r = self.res2_2E(xe2_1_r) 
 
        xf2_1 , xf2_1_r = self.res2_1F(xf1_2_r)
        xf2_2 , xf2_2_r = self.res2_2F(xf2_1_r)
        
        #======== third ResNet Block ==================
        xe3_1 , xe3_1_r = self.res3_1E(xe2_2_r)  #1/8
        xe3_2 , xe3_2_r = self.res3_2E(xe3_1_r)
 
        xf3_1 , xf3_1_r = self.res3_1F(xf2_2_r)
        xf3_2 , xf3_2_r = self.res3_2F(xf3_1_r)
        
        #======== fourth ResNet Block ==================
        xe4_1 , xe4_1_r = self.res4_1E(xe3_2_r)   #1/16
        xe4_2 , xe4_2_r = self.res4_2E(xe4_1_r)
 
        xf4_1 , xf4_1_r = self.res4_1F(xf3_2_r)
        xf4_2 , xf4_2_r = self.res4_2F(xf4_1_r)
        
        
        
        
        
        
        #============= Concatenate Event and Frame layers after ResNet Blocks ============================              
        c0 = xe                          # 1
        c1 = concatenate([xe1_2, xf])    # 1/2
        c2 = concatenate([xe2_2, xf1_2]) # 1/4
        c3 = concatenate([xe3_2, xf2_2]) # 1/8
        c4 = concatenate([xe4_2, xf3_2]) # 1/16
        

        #================ add conv 1x1 for Concatenate Event and Frame layers ============================
        C0 = self.conv1x1_0(c0)
        C1 = self.conv1x1_1(c1)
        C2 = self.conv1x1_2(c2)
        C3 = self.conv1x1_3(c3)
        C4 = self.conv1x1_4(c4)
        C5 = self.conv1x1_5(xf4_2_r)
        
        
        #======================== upsample part ==========================================================
        #==== frist upsample with only 1/32 scale =====
        #==== frist upsample layer x2 =================
        U5 = self.deconv5(C5)  # 512 , 1/16
        
        #====== frist upsample layer x4 ================
        U41 = concatenate([U5, C4]) # 512 + 512 
        U42 = self.convU4(U41)      # 512
        U4 = self.deconv4(U42)      # 512 , 1/16 >> 1/8


        #===== second upsample layer x8 ================
        U31 = concatenate([U4, C3]) # 512 + 256 
        U32 = self.convU3(U31)      # 256
        U3 = self.deconv3(U32)      # 256 >> 1/4
        
        
        #===== third upsample layer x16 ================
        U21 = concatenate([U3, C2]) # 256 + 128 
        U22 = self.convU2(U21)      # 128
        U2 = self.deconv2(U22)      # 128 >> 1/2
        
        #===== fourth upsample layer x32 ================
        U11 = concatenate([U2, C1]) # 128 + 64
        U12 = self.convU1(U11)      # 64
        U1 = self.deconv1(U12)      # 64 >> 1/1
        
        #================== final conv layer ========
        U01 = concatenate([U1, C0]) # 64 + 64
        U02 = self.convU0(U01)      # 64
        U00 = self.convU00(U02) # 64 >> 32
        out = self.FinalLayer(U00) # 32 >> 8
        
        return out
        
        
    
    
    
    
    #============ get graph of model ====================
    def build_graph(self):
        in1 = Input(shape=(128,256,8) , name = "input1")
        in2 = Input(shape=(256,512,8) , name = "input2") 
        inputs = [in1,in2]
        return Model(inputs=inputs, outputs=self.call(inputs))
    
 
  
'''
model = SegModel(8)
in1 = Input(shape=(128,256,8) , name = "input1")
in2 = Input(shape=(256,512,8) , name = "input2") 
model([in1,in2])


#============= plot model =====================
model.build_graph().summary()
# Just showing all possible argument for newcomer.  
tf.keras.utils.plot_model(
    model.build_graph(),                      # here is the trick (for now)
    to_file='model.png', dpi=96,              # saving  
    show_shapes=True, show_layer_names=True,  # show shapes and layer name
    expand_nested=False                       # will show nested block
)
'''

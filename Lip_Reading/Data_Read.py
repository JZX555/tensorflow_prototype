# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 20:55:34 2019

@author: 50568
"""


import tensorflow as tf #abc
import numpy as np
import Img

def parse_data(path, shape, size, batch_size, time_step):
    DataSet = []
    i_Buffers = []
    Labels = []
    l_Buffers = []
    cnt = 0
    
    for i in range(time_step):
        i_Buffers.append(None)
        l_Buffers.append(None)
        
    images = Img.Video_Read(path, shape, size)
    labels = Img.Label_Read(path)
    
    for image, label in zip(images, labels):
        cnt += 1
        for i in range(time_step):
            i_Buffers[time_step -i - 1] = i_Buffers[time_step - i - 2]            
            l_Buffers[time_step -i - 1] = l_Buffers[time_step - i - 2]
        i_Buffers[0] = image        
        l_Buffers[0] = label     
        
        if(cnt >= time_step):
            DataSet.append(i_Buffers.copy()) 
            Labels.append(l_Buffers.copy())
    
    train = tf.data.Dataset.from_tensor_slices((DataSet, Labels))
    train = train.batch(batch_size)
    
    return train
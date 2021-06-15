# -*- coding: utf-8 -*-
"""
Created on Thu May 13 17:34:25 2021

@author: onee
"""

import selectData
import numpy as np

from tensorflow.keras.utils import to_categorical

def chioceUnknown(dataset, data, unknown, num, selection):
    n_class = 10
    
    if selection == 'random':
        unknown = selectData.choiceRandom(dataset,num,unknown)
    elif selection == 'uniform':
        b = 20 # 이거 몇으로 설정할지 고민하자!
        unknown = selectData.choiceUniform(dataset,num,b,data,unknown,n_class)
    elif selection == 'topk':
        unknown = selectData.choiceTopk(dataset,num,unknown,n_class)
    elif selection == 'rtopk':
        unknown = selectData.choiceReverseTopk(dataset,num,unknown,n_class)
        
    return unknown

def unknownClassData(dataset, data, labels, unknown, num, selection):
    data = np.reshape(data,(len(data),28,28,1))
    unknown = np.reshape(unknown,(len(unknown),28,28,1))
    
    unknown = chioceUnknown(dataset,data,unknown,num,selection)
    
    data = np.vstack([data,unknown])
    labels = np.hstack([labels,np.array([10]*len(unknown))])
    
    import random
    r_data = list(zip(data, labels))
    random.shuffle(r_data)
    data, labels = zip(*r_data)
    data = np.array(data)
    labels = np.array(labels)
    
    data = data.astype('float32')
    data /= 255
    
    labels = np.array(to_categorical(labels,11))
        
    return data, labels

def hardsharingData(data, labels, unknown, num, selection):
    unknown = chioceUnknown(data,unknown,num,selection)
    return data, labels
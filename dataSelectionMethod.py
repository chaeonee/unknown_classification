# -*- coding: utf-8 -*-
"""
Created on Fri May 14 14:35:43 2021

@author: onee
"""

import pickle
import numpy as np

import prepareData
import model_struct

def trainModel(dataset, selection, epochs, batch_size):
    # load data for train
    if dataset == 'mnist':
        from keras.datasets import mnist
        (x_train, y_train), (_, _) = mnist.load_data()
        # (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        with open('./data/cifar10.p','rb') as file:
            unknown = pickle.load(file)
            _ = pickle.load(file)
            _ = pickle.load(file)
            _ = pickle.load(file)
        
    elif dataset == 'cifar10':
        with open('./data/cifar10.p','rb') as file:
            x_train = pickle.load(file)
            y_train = pickle.load(file)
            _ = pickle.load(file)
            _ = pickle.load(file)
            
        from keras.datasets import mnist
        (unknown, _), (_, _) = mnist.load_data()
    
    with open('./data/emnist.p','rb') as file:
        emnist = pickle.load(file)
        _ = pickle.load(file) 
    with open('./data/imagenet.p','rb') as file:
        imagenet = pickle.load(file)
        _ = pickle.load(file)
        _ = pickle.load(file)
        _ = pickle.load(file)
    
    unknown = np.vstack([unknown,emnist,imagenet])
    
    num = 5000
    x_train, y_train = prepareData.unknownClassData(dataset,x_train,y_train,unknown,num,selection)
    
    model = model_struct.MultiCls(x_train[0],11)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    
    model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size)
    
    model.save('./model/test2/'+dataset+'_dataselection_'+selection+'.h5')

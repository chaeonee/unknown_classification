# -*- coding: utf-8 -*-
"""
Created on Fri May 14 14:34:11 2021

@author: onee
"""

import pickle
import numpy as np

import prepareData
import model_struct

def trainModel(num, epochs, batch_size, dataset):
    # load data for train
    if dataset == 'mnist':
        from keras.datasets import mnist
        (x_train, y_train), (_, _) = mnist.load_data()
        
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
    
    x_train, y_train = prepareData.unknownClassData(x_train,y_train,unknown,num,'random')
    
    model = model_struct.MultiCls(x_train[0],11)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    
    model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size)
    
    model.save('./model/test1/'+dataset+'_unknownNetwork_'+str(num)+'.h5')
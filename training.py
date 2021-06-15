# -*- coding: utf-8 -*-
"""
Created on Thu May 13 17:42:25 2021

@author: onee
"""

import argparse

from numOfUnknown import trainModel as t1
from dataSelectionMethod import trainModel as t2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', type=int, required=False, default=300)
    parser.add_argument('--batch_size', type=int, required=False, default=128)
    parser.add_argument('--dataset', required=False, default='mnist')
    parser.add_argument('--n', type=int, required=True, help='Test Number')
    
    args = parser.parse_args()
    
    if args.n == 0:
        # Train pre-train model
        import numpy as np
        import model_struct
        from tensorflow.keras.utils import to_categorical
        
        if args.datset == 'mnist':
            from keras.datasets import mnist
            (x_train, y_train), (_, _) = mnist.load_data()
        
        elif args.dataset == 'cifar10':
            import pickle
            with open('./data/cifar10.p','rb') as file:
                x_train = pickle.load(file) 
                y_train = pickle.load(file)    
                #_ = pickle.load(file)
                #_ = pickle.load(file)
                
        x_train = np.reshape(x_train,(len(x_train),28,28,1))
        x_train = x_train.astype('float32')
        x_train /= 255
        y_train = np.array(to_categorical(y_train,10))
        model = model_struct.MultiCls(x_train[0],10)
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        model.fit(x_train,y_train,epochs=args.epochs,batch_size=args.batch_size)
        model.save('./model/'+args.dataset+'Network_epoch'+str(args.epochs)+'.h5')
        
    elif args.n == 1:
        # Test1 training
        for num in [500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]:
            t1(num,args.epochs,args.batch_size,args.dataset)
            
    elif args.n == 2:
        #Test2 training
        for s in ['random','topk','rtopk','uniform']:
            t2(args.dataset,s,args.epochs,args.batch_size)
            
    elif args.n == 3:
        # Test3 training
        # if you want to train own model, you can train the model in 'SingleModel.py'
        # you can ensemble With pre-saved models
        print('')
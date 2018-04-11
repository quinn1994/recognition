# -*- coding: utf-8 -*-
#!/usr/bin/env python3
#author:quinn lee
#date: 2018-04-01
import numpy as np
import pickle
import gzip
def load_data():
    f=gzip.open('/Users/Administrator/Documents/GitHub/recognition/data/mnist.pkl.gz','rb')
    training_data,validation_data,test_data=pickle.load(f,encoding="bytes")
    f.close()
    return (training_data,validation_data,test_data)

def load_data_wrapper():
    tr_d,va_d,te_d=load_data()
    '''read JSON data'''
    training_inputs=[np.reshape(x,(784,1)) for x in tr_d[0]]
    training_results=[vectorized_result(y) for y in tr_d[1]]
    training_data=list(zip(training_inputs,training_results))

    validation_inputs=[np.reshape(x,(784,1)) for x in va_d[0]]
    validation_data=list(zip(validation_inputs,va_d[1]))

    test_inputs=[np.reshape(x,(784,1)) for x in te_d[0]]
    test_data=list(zip(test_inputs,te_d[1]))

    return (training_data,validation_data,test_data)

def vectorized_result(j):
    e=np.zeros((10,1))
    e[j]=1.0
    return e
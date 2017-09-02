#coding:utf-8
'''
Created on 2017/9/2 上午9:45

@author: liucaiquan
'''

import numpy as np
def makeFeatureVec(words, model, num_feature):
    featureVec=np.zeros((num_feature,),dtype="float32")
    nwords=0
    index2word_set=set(model.index2word)

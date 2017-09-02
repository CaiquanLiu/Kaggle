#coding:utf-8
'''
Created on 2017/9/2 下午4:12

@author: liucaiquan
'''
import pandas as pd
train=pd.read_csv('../datasets/train.csv')
print train.shape

test=pd.read_csv('../datasets/test.csv')
print test.shape
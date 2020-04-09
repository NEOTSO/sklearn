#!/usr/bin/python
# -*- coding: utf-8 -*-

# 特征预处理-归一化

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def minmax():
    data = pd.read_csv('dating.txt')
    data = data.iloc[:, :3]
    print(data)
    print(type(data))
    transfer = MinMaxScaler()
    data_new = transfer.fit_transform(data)
    print(data_new)
    print(type(data_new))

if (__name__ == '__main__'):
    minmax()
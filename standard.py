#!/usr/bin/python
# -*- coding: utf-8 -*-

# 特征预处理-标准化

import pandas as pd
from sklearn.preprocessing import StandardScaler

def standard():
    data = pd.read_csv('dating.txt')
    data = data.iloc[:, :3]
    print(data)
    print(type(data))
    transfer = StandardScaler()
    data_new = transfer.fit_transform(data)
    print(data_new)
    print(type(data_new))

if (__name__ == '__main__'):
    standard()
#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr

def variance():
    data = pd.read_csv('factor_returns.csv')
    data = data.iloc[:, 1:-2]
    print(data)
    # threshold: 指定阈值方差
    transform = VarianceThreshold(threshold=5)
    data_new = transform.fit_transform(data)
    print(data_new, data_new.shape)

    r1 = pearsonr(data['pe_ratio'], data['pb_ratio'])
    print('相关系数1:', r1)

    r2 = pearsonr(data['revenue'], data['total_expense'])
    print('相关系数1:', r2)

if (__name__ == '__main__'):
    variance()
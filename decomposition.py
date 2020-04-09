#!/usr/bin/python
# -*- coding: utf-8 -*-

# pca降维

from sklearn.decomposition import PCA

def pca_demo():
    data = [
        [2,8,4,5],
        [6,3,0,8],
        [5,4,9,1]
    ]

    # 保留2个特征信息
    #transfer = PCA(n_components=2)

    # 保留95%信息
    transfer = PCA(n_components=0.95)
    data_new = transfer.fit_transform(data)
    print(data_new)


if (__name__ == '__main__'):
    pca_demo()
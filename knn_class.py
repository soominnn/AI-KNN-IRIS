#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import collections

#knn 클래스 생성
#k,X,y,y_names과 같은 변수들은 main에 구현하였습니다.
class Knn:
    
    #테스트케이스와의 거리 계산하는 distance 함수
    def distance(X_test,X_train):
        d = 0
        for i in range(4):
            d += np.power(X_test[i]-X_train[i],2)
        d = np.sqrt(d)    
        return d
    
    
    #k개의 이웃한 neighbor 구해서 정렬하는 K-Nearest Neighbor 함수
    def neighbor(y,k,distance_memo_index):
        sort_target = np.zeros(k)
        for m in range(k):
            sort_target[m] = y[distance_memo_index[m+1]]
        return sort_target
    
    
    #가까운 데이터중에서 갯수가 많은 데이터 결정 함수
    def majority(list_target):
        cnt = collections.Counter(list_target)
        target_major = cnt.most_common(1)[0][0]
        return target_major
    
    
    #최소거리부터 거리에 따르는 가중치를 더한 weighted major vote함수
    def weighted_majority(k, list_target):
        weighted = [0] * k
        for a in range(k):
            temp = int(list_target[a])
            weighted[temp] += k-a
        tmp = max(weighted)
        print(weighted)
        return weighted.index(tmp)


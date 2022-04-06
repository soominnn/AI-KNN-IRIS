#!/usr/bin/env python
# coding: utf-8

# In[1]:


import import_ipynb
import numpy as np
import matplotlib.pyplot as plt
from knn_class import Knn
from sklearn.datasets import load_iris
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

#아이리스 꽃 데이터 추출
iris = load_iris()
X = iris.data 
y = iris.target
y_name = iris.target_names


#k개의 neighbor를 구하기 위한 입력
k = int(input())


#테스트케이스를 위한 변수 선언
distance_memo = np.zeros(150)
test_case = []

#결과 정확도를 위한 변수 선언
probability = 0
weighted_probability = 0

#2차원 데이터 시각화
x1_min, x1_max = X[:, 0].min() - .5, X[:, 0].max() + .5
x2_min, x2_max = X[:, 1].min() - .5, X[:, 1].max() + .5
plt.figure(figsize=(8, 6))
# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
 edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.show()

#3차원 데이터 시각화
fig = plt.figure(figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])
plt.show()


#15간격으로 테스트케이스 저장해두기
for i in range(len(X)):
    if (i+1) % 15 == 0:
        test_case.append(X[i])



for i in range(len(test_case)):
    
    #테스케이스와의 거리 distance 함수로 구하기
    for j in range(len(X)):
        distance_memo[j] = Knn.distance(test_case[i], X[j])
    
    
    #최소거리를 오름차순으로 정렬
    #첫번째 인덱스는 테스트케이스 자신이여서 이후에 계산시에는 인덱스 1부터 시작
    distance_memo_index = np.argsort(distance_memo)    
    
    
    #테스트케이스와의 거리 출력
    print(distance_memo)
    
    
    #최소거리를 오름차순으로 출력
    #첫번째 인덱스는 테스트케이스 자신이여서 이후에 계산시에는 인덱스 1부터 시작
    print(distance_memo_index) 
    
    
    
    #k개의 이웃한 neighbor 구해서 정렬하는 K-Nearest Neighbor 함수
    list_target = Knn.neighbor(y,k,distance_memo_index)
    print(list_target)
    
    #정렬된 가운데 majority 함수를 이용해 가장 많이 이웃한 데이터 구하기
    major_target = Knn.majority(list_target)
    

    
    #최소거리부터 거리에 따르는 가중치를 더한 weighted major vote함수
    weighted_major_target = Knn.weighted_majority(k, list_target)
    
    
    
    #위에서 구한 데이터에 따르는 꽃 출력,비교(majority vote방식)
    if major_target == 0:
        major_target_name = "setosa"
    elif major_target == 1:
        major_target_name = "versicolor"
    elif major_target == 2:
        major_target_name = "virginica"
    print("-majority vote-")
    print(f"Test Data Index: {i} Computed class: {major_target_name}, True class: {y_name[y[15*(i+1)-1]]}")
    if major_target_name == y_name[y[15*(i+1)-1]]:
        probability +=1
        
        
    #위에서 구한 데이터에 따르는 꽃 출력,비교(weighted majority vote방식)
    if weighted_major_target == 0:
        weighted_major_target_name = "setosa"
    elif weighted_major_target == 1:
        weighted_major_target_name = "versicolor"
    elif weighted_major_target == 2:
        weighted_major_target_name = "virginica"
    print("-weighted majority vote-")
    print(f"Test Data Index: {i} Computed class: {weighted_major_target_name}, True class: {y_name[y[15*(i+1)-1]]}")
    if weighted_major_target_name == y_name[y[15*(i+1)-1]]:
        weighted_probability +=1


#정확도 결과 출력
print("\n")     
print("============================result============================")
print(f"-majoritty vote 정확도: {probability/10}")
print(f"-majoritty vote 확률: {(probability/10)*100}%")
print(f"-weighted majoritty vote 정확도: {weighted_probability/10}")
print(f"-weighted majoritty vote 확률: {(weighted_probability/10)*100}%")


# In[ ]:





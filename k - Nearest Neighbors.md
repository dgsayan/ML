---
layout: page
title: "K Nearest Neighbors"
permalink: /knn/
---



## The k-NN algorithm

### 1. Concept

1.1 <ins>Assumption</ins>: Similar Inputs have similar outputs

1.2 <ins>Classification rule</ins>: For a test input $x$, the most common (mode) label among its $k$ most similar training samples is returned as its output label

1.3 <ins>Formal definition</ins>:

- Test point : $x$
- Denote the set of $k$ nearest neighbors as $S_x$, such that $S_x \in D$ and $|S_x| = k$. Then ∀ $(x',y') \in D\setminus{}S_x$ and ∀ $(x'',y'') \in S_x$,

$$ dist(x, x') >= max (dist(x, x'')) $$

- That is, every point in $D$ is at least as far away from $x$ as the farthest point in $S_x$ 
- We can then define the classifier $h()$ as a function returning the most common label in $S_x$

$$ h(x) = mode({y'' : (x'',y'') \in S_x}) $$


1.4 <ins>Distance Metric</ins>:

The k-NN classifier fundamentally relies on a distance metric to identify the nearest neighbors. The better that metric quantifies similarity, the better the classification. The most common choice is the **Minkowski Distance**

$$ dist(x, z) = (\displaystyle\sum_{i=1}^{d}|x_i-z_i|^{p}) ^{1/p} $$

- p = 1 gives us Manhattan distance 
- p = 2 gives us Eucledian distance


### 2. Pseudocode

$\textrm{Classify}(\mathbf{X,Y},x)\;//\;\mathbf{X}: m\;\text{rows of training data},\;\mathbf{Y}:\text{class labels of}\;\mathbf{X},\;x:\text{test point}$

$\mathbf{for}\;i = 1\;\text{to}\;m\;\mathbf{do}\\\hspace{3ex}\text{Compute distance}\;d(\mathbf{X}_{i},x)\newline\mathbf{end\;for}$

$\text{Compute set}\;I\;\text{containing indices for the}\;k\;\text{smallest distances in}\;d(\mathbf{X}_{i},x)\\\mathbf{return}\;\text{majority class label for}\;\{\mathbf{Y_i},\;\text{where}\;i\in I\}$

### 3. Application

The Iris data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant.
<br>Predicted attribute : class of Iris plant
<br>Information attributes : sepal length, sepal width, petal length, petal width

In the following section, we build a KNN classifier in python and make predictions for some test points


```python
# Import libraries

import os
import numpy as np
```


```python
# Load data

X = np.genfromtxt('datasets\iris.csv', delimiter = ',', dtype = None, skip_header = 1, encoding = 'UTF-8', usecols = [0,1,2,3])
Y = np.genfromtxt('datasets\iris.csv', delimiter = ',', dtype = None, skip_header = 1, encoding = 'UTF-8', usecols = 4)
```


```python
X.shape
```




    (150, 4)




```python
Y.shape
```




    (150,)




```python
# Split data into train-test

np.random.seed(0)
ids = np.arange(Y.shape[0])
test_ids = np.random.choice(ids, size = 5, replace = False)
train_ids = ids[~np.isin(ids, test_ids)]

X_train = X[train_ids]
Y_train = Y[train_ids]
x_test = X[test_ids]
y_test = Y[test_ids]
```


```python
# Minkowski distance

def minkowski_dist(v1, v2, p):
    return np.power(np.sum(np.power(abs(v1-v2), p)), 1/p)

# Mode 1-d

def mode1d(v):
    vals, cnt = np.unique(v, return_counts = True)
    return vals[cnt.argmax()]

```


```python
# kNN

def kNN_classify(X, Y, x, p, k):
    dist_arr = np.fromiter((minkowski_dist(v, x, p) for v in X), 'float')
    idx_k_smallest = np.argpartition(dist_arr, k)[:k]
    return mode1d(Y[idx_k_smallest])

```


```python
# prediction

pred = np.fromiter((kNN_classify(X_train, Y_train, x, 2, 3) for x in x_test), '<U12')

flag = pred == y_test

score = sum(flag)/len(flag)

score
```




    1.0



### 3. Miscellaneous

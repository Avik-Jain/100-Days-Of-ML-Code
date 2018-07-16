# K-Nearest Neighbors (K-NN)

<p align="center">
  <img src="https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Info-graphs/Day%207.jpg">
</p>

## The DataSet | Social Network 

<p align="center">
  <img src="https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Other%20Docs/data.PNG">
</p> 


## Importing the libraries
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

## Importing the dataset
```python
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
```

## Splitting the dataset into the Training set and Test set
```python
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
```
## Feature Scaling
```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```
## Fitting K-NN to the Training set
```python
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
```
## Predicting the Test set results
```python
y_pred = classifier.predict(X_test)
```

## Making the Confusion Matrix
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
```

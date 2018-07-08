# Multiple Linear Regression


<p align="center">
  <img src="https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Info-graphs/Day%203.jpg">
</p>


## Step 1: Data Preprocessing

### Importing the libraries
```python
import pandas as pd
import numpy as np
```
### Importing the dataset
```python
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : ,  4 ].values
```

### Encoding Categorical data
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[: , 3] = labelencoder.fit_transform(X[ : , 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
```

### Avoiding Dummy Variable Trap
```python
X = X[: , 1:]
```

### Splitting the dataset into the Training set and Test set
```python
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
```
## Step 2: Fitting Multiple Linear Regression to the Training set
```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
```

## Step 3: Predicting the Test set results
```python
y_pred = regressor.predict(X_test)
```

# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM: 
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Load and Preprocess Data: Read the dataset, drop unnecessary columns, and convert categorical variables into numerical codes using .astype('category') and .cat.codes.

2.Define Variables: Split the dataset into features (X) and target variable (Y), and initialize a random parameter vector theta.

3.Implement Functions: Define the sigmoid, loss, gradient_descent, and predict functions for logistic regression.

4.Train Model: Use gradient descent to optimize the parameters theta over a specified number of iterations.

5.Evaluate and Predict: Calculate accuracy of predictions on the training data, and demonstrate predictions with new sample data.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: THAMIZH KUMARAN
RegisterNumber:  212223240166
*/

import pandas as pd
import numpy as np
df = pd.read_csv('Placement_Data.csv')
df

df = df.drop('sl_no',axis=1)
df = df.drop('salary', axis=1)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["gender"]=df["gender"].astype('category')
df["ssc_b"]=df["ssc_b"].astype('category')
df["hsc_b"]=df["hsc_b"].astype('category')
df["hsc_s"]=df["hsc_s"].astype('category')
df["degree_t"]=df["degree_t"].astype('category')
df["workex"]=df["workex"].astype('category')
df["specialisation"]=df["specialisation"].astype('category')
df["status"]=df["status"].astype('category')
df.dtypes

df["gender"]=df["gender"].cat.codes
df["ssc_b"]=df["ssc_b"].cat.codes
df["hsc_b"]=df["hsc_b"].cat.codes
df["hsc_s"]=df["hsc_s"].cat.codes
df["degree_t"]=df["degree_t"].cat.codes
df["workex"]=df["workex"].cat.codes
df["specialisation"]=df["specialisation"].cat.codes
df["status"]=df["status"].cat.codes
df

X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values
Y

theta = np.random.random(X.shape[1])
y=Y

def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*log(1-h))

def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient
    return theta

theta = gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)

def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred

y_pred = predict(theta, X)

accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy:", accuracy)

print(y_pred)

print(y)

xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)

xnew = np.array([[0, 0 , 0, 0, 0, 2, 8, 2, 0, 0, 1, 0,]])
y_prednew = predict(theta, xnew)
print(y_prednew)

```

## Output:

### Dataset
![Screenshot 2025-03-25 182054](https://github.com/user-attachments/assets/32e93d00-ca4a-4e48-baef-07f63a367b53)

### Dataset.dtypes
![Screenshot 2025-03-25 182101](https://github.com/user-attachments/assets/e795ee83-67a5-4553-8bd9-e489206db06b)

### Dataset
![Screenshot 2025-03-25 182111](https://github.com/user-attachments/assets/2cd1618e-9ba5-4b82-86ea-65c13df7f87e)

### Y
![Screenshot 2025-03-25 182117](https://github.com/user-attachments/assets/b0f98dbe-27e0-4e53-adf6-e369bca5a1cf)

### Accuracy
![Screenshot 2025-03-25 182125](https://github.com/user-attachments/assets/e036147d-6bac-416e-86cf-e4083f85fa31)

### Y_pred
![Screenshot 2025-03-25 182139](https://github.com/user-attachments/assets/7e34cbaf-86e7-4030-8f19-6b0cef37e769)

### Y
![Screenshot 2025-03-25 182147](https://github.com/user-attachments/assets/948edb17-b258-448b-be3d-1e5e56f2c3e4)

### Y_prednew
![Screenshot 2025-03-25 182154](https://github.com/user-attachments/assets/cb38dbf3-e69e-4f70-885d-77dbe6648648)

### Y_prednew
![Screenshot 2025-04-17 140518](https://github.com/user-attachments/assets/ade28014-2b1e-4f54-88bd-9cebe0b82cee)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.


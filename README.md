# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Initialize Parameters – Set initial values for slope m and intercept 𝑏 and choose a learning rate 𝛼

2.Compute Cost Function – Calculate the Mean Squared Error (MSE) to measure model performance.

3.Update Parameters Using Gradient Descent – Compute gradients and update m and b using the learning rate.

4.Repeat Until Convergence – Iterate until the cost function stabilizes or a maximum number of iterations is reached.



## Program & Output:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: THIRISHA A
RegisterNumber:212223040228
*/

import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler

def linear_regression(X1, y, learning_rate=0.1, num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        
        errors=(predictions-y).reshape(-1,1)

        theta -=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta

data=pd.read_csv("/content/50_Startups.csv")
data.head()
```
![423053582-0af5a9f5-66cc-4d99-9e59-78135fb527d7](https://github.com/user-attachments/assets/07cb98cc-764f-4eba-bbed-16994cb5802a)

```
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
```
![423053601-7d2fca11-8177-4239-9507-4213263c8dc3](https://github.com/user-attachments/assets/ba42ca60-84cb-4987-87a2-fb2ca6ad55d5)

```
print(X1_Scaled)
```

![423053624-364ebe3d-87d7-4a53-bf22-92994fb9bfcc](https://github.com/user-attachments/assets/e47c8fb9-89dc-4cd5-9568-d836e1f742e6)
```
theta=linear_regression(X1_Scaled, Y1_Scaled)
new_data= np.array([165349.2 , 136897.8 , 471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1, new_scaled), theta)
prediction= prediction.reshape(-1,1)
pre = scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```

![423053650-ff1329fe-0b32-4695-b7a3-a551489c961a](https://github.com/user-attachments/assets/d48d5e15-a110-42eb-ba7c-3444524082e5)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.

# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the standard Libraries.
 2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
 4.Assign the points for representing in the graph.
 5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
```

## Program:
```
/*
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: GOLLA MOULOIDHAR
RegisterNumber:  212223240042
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)  
*/
```

## Output:
df.head()
![image](https://github.com/user-attachments/assets/66982678-6d70-4f5f-aa7a-bf91439c9845)
df.tail()
![image](https://github.com/user-attachments/assets/f9f5414f-0616-41b0-9084-9e0ec16c936c)
Array value of X
![image](https://github.com/user-attachments/assets/2eb422d9-5419-4133-bbcc-682d17ad3bf3)
Array value of Y
![image](https://github.com/user-attachments/assets/a5c297dd-552c-4dac-96c2-c0bdff464c3c)
![image](https://github.com/user-attachments/assets/3cc49faa-b484-4f4d-bb1f-66a419b1f51f)
Array values of Y test
![image](https://github.com/user-attachments/assets/fc9272f5-37ac-435f-afd9-f736ff9efea3)
Training Set Graph
![image](https://github.com/user-attachments/assets/3993fc04-a4f9-4e1e-b4ff-c162d2ffa236)
Test Set Graph
![image](https://github.com/user-attachments/assets/e36ae6ec-1cc5-4223-a349-3089d876d872)








## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

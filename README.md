# BLENDED LEARNING
# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries such as pandas, sklearn, seaborn, and matplotlib.
2. Load the dataset food_items_binary.csv.
3. Select important features (Calories, Total Fat, Saturated Fat, Sugars, Dietary Fiber, Protein).
4. Define the target variable (class).
5. Split the dataset into training and testing sets.
6. Apply StandardScaler to normalize the feature values and Create an SVM classifier (SVC) model.
7. Define a parameter grid for hyperparameter tuning.
8. Use GridSearchCV with 5-fold cross validation to find the best parameters.
9. Train the model using the training dataset and Predict the output for the test dataset.
10. Calculate accuracy and generate a classification report.
11. Display the confusion matrix using a heatmap.

## Program:
```
/*
Program to implement SVM for food classification for diabetic patients.
Developed by: POPURI SAHITHYA
RegisterNumber:  212225240106
*/

import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv('food_items_binary.csv')
print(data.head())
print(data.columns)
features=['Calories','Total Fat','Saturated Fat','Sugars','Dietary Fiber','Protein']
target='class'
X=data[features]
y=data[target]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
svm=SVC()
param_grid={
    'C':[0.1,1,10,100],
    'kernel':['linear','rbf'],
    'gamma':['scale','auto']
}
grid_search=GridSearchCV(svm,param_grid,cv=5,scoring='accuracy')
grid_search.fit(X_train,y_train)
best_model=grid_search.best_estimator_
print("Name: POPURI SAHITHYA")
print("Register number: 212225240106")
print("Best parameters:",grid_search.best_params_)
y_pred=best_model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print("Name:POPURI SAHITHYA")
print("Register number: 212225240106")
print("Accuracy:",accuracy)
print("Classification Report:\n",classification_report(y_test,y_pred))
conf_matrix=confusion_matrix(y_test,y_pred)
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

## Output:

<img width="795" height="703" alt="image" src="https://github.com/user-attachments/assets/ba862647-342c-47c1-bb3f-9f3469581806" />

<img width="675" height="335" alt="image" src="https://github.com/user-attachments/assets/3d6a9bec-7a97-4249-bbde-d0770733c940" />

<img width="708" height="576" alt="image" src="https://github.com/user-attachments/assets/b30a2570-ad42-499e-a520-2b83325f3ccb" />

## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.

# -*- coding: utf-8 -*-
"""
Created on Sun May 31 17:24:52 2020

@author: Bibhusmita
"""
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:37:57 2020

@author: Bibhusmita
"""
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("student-math.csv", sep=";") 

df.isnull().sum()


#creating final_grade column
df["final_grade"] = df["G1"] + df["G2"] + df["G3"]


#Label encoding
encode = LabelEncoder()

df["school"] =  encode.fit_transform(df["school"])
df["sex"] =  encode.fit_transform(df["sex"])
df["address"] = encode.fit_transform(df["address"])
df["famsize"] = encode.fit_transform(df["famsize"])
df["Pstatus"] =  encode.fit_transform(df["Pstatus"])
df["schoolsup"] =  encode.fit_transform(df["schoolsup"])
df["famsup"] = encode.fit_transform(df["famsup"])
df["paid"] = encode.fit_transform(df["paid"])
df["activities"] =  encode.fit_transform(df["activities"])
df["nursery"] =  encode.fit_transform(df["nursery"])
df["higher"] = encode.fit_transform(df["higher"])
df["internet"] = encode.fit_transform(df["internet"])
df["romantic"] =  encode.fit_transform(df["romantic"])

#encoding nominal values
df["Mjob"] =  encode.fit_transform(df["Mjob"])
df["Fjob"] = encode.fit_transform(df["Fjob"])
df["reason"] = encode.fit_transform(df["reason"])
df["guardian"] =  encode.fit_transform(df["guardian"])
ct = ColumnTransformer([('encoder',OneHotEncoder(),['Mjob','Fjob','reason','guardian'])], remainder = 'passthrough')
df = np.array(ct.fit_transform(df), dtype = np.int64)




#creating array y
y = df[:,-1]
x = df[:,:-2]

#data splitting
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

#------------------------------Task2-----------------------------------------------------

#creating a linear regression model and predicting
from sklearn import linear_model
model_linear = linear_model.LinearRegression()
model_linear.fit(x_train, y_train)
y_pred = model_linear.predict(x_test)

#creating a RandomForestRegressor model and predicting
from sklearn.ensemble import RandomForestRegressor
model_random = RandomForestRegressor(n_estimators = 300)
model_random.fit(x_train,y_train)
y_predr = model_random.predict(x_test)


from sklearn.tree import DecisionTreeRegressor
model_decision = DecisionTreeRegressor()
model_decision.fit(x_train,y_train)
y_predd = model_decision.predict(x_test)


#predicting and calculating the accuracy scores
print("Linear regression model:\n-------------------------------")
print("Train score:",model_linear.score(x_train,y_train))
print("Test score:",model_linear.score(x_test,y_test))
print("Predict score:",model_linear.score(x_test,y_pred))
plt.scatter(y_test,y_pred,color ='red')
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.title("y_test vs y_pred(Linear Regression")
plt.show()

print("RandomForestRegressor model:\n-------------------------------")
print("Train score:",model_random.score(x_train,y_train))
print("Test score:",model_random.score(x_test,y_test))
print("Predict score:",model_random.score(x_test,y_predr))
fig1 = plt.scatter(y_test,y_predr, color ="red")
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.title("y_test vs y_pred(RandomForestRegressor)")
plt.show()

print("DecisionTreeRegressor model:\n-------------------------------")
print("Train score:",model_decision.score(x_train,y_train))
print("Test score:",model_decision.score(x_test,y_test))
print("Predict score:",model_decision.score(x_test,y_predd))
plt.scatter(y_test,y_predd, color ="red")
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.title("y_test vs y_pred(DecisionTreeRegressor)")
plt.show()



#plotting the scatterplot
fig, ax = plt.subplots(figsize = (10,5))
ax.scatter(y_test,y_pred, label = "Linear")
ax.set_xlabel("y_test")
ax.set_ylabel("y_pred")
ax.set_title("y_test vs y_pred\n Comparison")
ax.scatter(y_test,y_predr, color ='yellow',label ="RandomForest")
ax.scatter(y_test,y_predd,color = 'red', label = "DecisionTree")
ax.legend()
fig.savefig("Comparision_plot.png")




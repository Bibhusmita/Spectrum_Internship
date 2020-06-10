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
from sklearn import linear_model
import matplotlib.pyplot as plt

df = pd.read_csv("student-math.csv", sep=";") 

df.isnull().sum()


#creating final_grade column
df["final_grade"] = df["G1"] + df["G2"] + df["G3"]

dfg = df.copy()


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

#creating a linear regression model
model = linear_model.LinearRegression()

#fitting the model
model.fit(x_train, y_train)

#predicting and calculating the accuracy score
y_pred = model.predict(x_test)
print("Linear regression model:\n-------------------------------")
print("Train score:",model_linear.score(x_train,y_train))
print("Test score:",model_linear.score(x_test,y_test))
print("Predict score:",model_linear.score(x_test,y_pred))

#plotting the scatterplot
plt.scatter(y_test,y_pred,color="red")
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.title("y_test vs y_pred(Linear Regression)")
plt.show()



#Building Optimal model using backward elimination model 
import statsmodels.api as sm

def backwardElimination(x,sl):
    colnums = len(x[0])
    for i in range(0,colnums):
        regressor_model = sm.OLS(y,x).fit()
        maxcol = max(regressor_model.pvalues.astype(float))
        if maxcol > sl:
            for j in range(0,colnums-i):
                if regressor_model.pvalues[j].astype(float) == maxcol :
                    x = np.delete(x,j,1)
    print(regressor_model.summary())        
                    
                
        


x_opt =x [:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,21,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44]]
sl = 0.05
backwardElimination(x_opt,sl)

#Data visualisation using seaborn

import seaborn as sns
sns.catplot(x='absences', y='final_grade', data = dfg, kind = 'bar' ,height = 3, aspect =2)
sns.catplot(y= 'absences',x="sex", data=dfg, kind='violin')
sns.swarmplot(x='sex', y='absences', color='k',size=3, data = dfg)
sns.catplot(x='studytime', y='final_grade', hue='sex', kind='violin',data =dfg)
sns.catplot(x='G1', y= 'final_grade', hue='sex',kind='bar', data=dfg)
sns.catplot(x='G2', y= 'final_grade', hue='sex',kind='bar', data=dfg)

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


#Step-1
df_csv = pd.read_csv("student-math.csv")
#The csv file provided to me, had all the data in one column with each data separated by ";"
col = df_csv.columns.tolist()
cols = col[0].split(";")
df_csv.columns=["data"]
df= df_csv.data.str.split(";",expand = True)
df.columns = cols
#len(df.columns.tolist())


#Step-2
df['G1'] = df['G1'].map(lambda x: x.strip('"'))
df['G2'] = df['G2'].map(lambda x: x.strip('"'))
df['final_grade'] = df['G1'].astype(int) + df['G2'].astype(int) + df['G3'].astype(int)


#Step-3
df = df.drop(['G1','G2','G3'],axis = 1)


#Step-4
df['school'] = df['school'].apply({'"GP"':0,'"MS"':1}.get)
df['sex'] = df['sex'].apply({'"F"':0,'"M"':1}.get)
df['address'] = df['address'].apply({'"R"':0,'"U"':1}.get)
df['famsize'] = df['famsize'].apply({'"LE3"':0,'"GT3"':1}.get)
df['Pstatus'] = df['Pstatus'].apply({'"A"':0,'"T"':1}.get)
df['schoolsup'] = df['schoolsup'].apply({'"No"':0,'"Yes"':1}.get)
df['famsup'] = df['famsup'].apply({'"No"':0,'"Yes"':1}.get)
df['paid'] = df['paid'].apply({'"No"':0,'"Yes"':1}.get)
df['activities'] = df['activities'].apply({'"No"':0,'"Yes"':1}.get)
df['nursery'] = df['nursery'].apply({'"No"':0,'"Yes"':1}.get)
df['higher'] = df['higher'].apply({'"No"':0,'"Yes"':1}.get)
df['internet'] = df['internet'].apply({'"No"':0,'"Yes"':1}.get)
df['romantic'] = df['romantic'].apply({'"No"':0,'"Yes"':1}.get)


#print(df.head())


#Step-5 and 6

#using seaborn
import seaborn as sns
ax = sns.boxplot(x='studytime',y='final_grade', data = df).set_title("Box_plot along with scatter points")
ax = sns.swarmplot(x='studytime' ,y='final_grade', data = df, color = "grey")

#using matplotlib and pandas
fig, ax = plt.subplots()
x = np.array(df['studytime'])
y = np.array(df['final_grade'])
group = df['studytime']
for g in np.unique(group):
    i = np.where(group == g)
    ax.scatter(x[i],y[i], label = g)
plt.legend(loc = "upper right", bbox_to_anchor=(1.15,1))
plt.title("Scatter plot for studytime vs final_grade")
plt.xlabel("studytime")
plt.ylabel("final_grade")
plt.show()

bp = df.boxplot(column = 'final_grade', by='studytime')
plt.suptitle('Box plot for studytime vs ')
bp.set_ylabel("final_grade")





import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


train_df = pd.read_csv('metrics data.csv')


#Accuracy plot***************************************************************************************************************
target1 = "accuracy"

# x = train_df.drop(target, axis=1).values
y = train_df[[target1]].values
trees = train_df[["trees"]].values
model = train_df[["model"]].values

trees.ravel()
model.ravel()
y=y.ravel() #Converting to 1D array
# print(model)
# av5 = 

# av10 = 

# av25 =

# av50 =
yensemble= [0]*4
ytakeda=[0]*4
for i in range(0,80):
    if(trees[i] == 5  and model[i] == 'takeda  '):
        ytakeda[0] += y[i] 
        # print(y[i])
    if(trees[i] == 5 and model[i] == "ensemble  " ):
        yensemble[0] += y[i]
    if(trees[i] == 10 and model[i] == "takeda  " ):
        ytakeda[1] += y[i] 
    if(trees[i] == 10 and model[i] == "ensemble  " ):
        yensemble[1] += y[i]
    if(trees[i] == 25 and model[i] == "takeda  " ):
        ytakeda[2] += y[i] 
    if(trees[i] == 25 and model[i] == "ensemble  " ):
        yensemble[2] += y[i]
    if(trees[i] == 50 and model[i] == "takeda  "):
        ytakeda[3] += y[i] 
    if(trees[i] == 50 and model[i] == "ensemble  "):
        yensemble[3] += y[i]
    
print(ytakeda)
print(yensemble)
ytakeda = np.array(ytakeda)
yensemble = np.array(yensemble)
ytakeda = ytakeda/10
yensemble = yensemble/10
# ytakeda= ytakeda/10
# print(ytakeda)

# yensemble=[80.34,79.48,82.05]
# ytakeda=[77.77,76.92]
xx=[5,10,25,50]
GPU1=plt.plot(xx,yensemble,label='ensemble approach')
plt.plot(xx,ytakeda,label='takeda approach')
# plt.scatter(xx,yta)
# plt.scatter(xx,yy1)
plt.legend()
plt.title("Metrics Analysis")
plt.ylabel("Accuracy%")
plt.xlabel("No of trees")
plt.show()


#Time plot***************************************************************************************************************

target2 = "time(s)"




# x = train_df.drop(target, axis=1).values
y = train_df[[target2]].values
trees = train_df[["trees"]].values
model = train_df[["model"]].values

trees.ravel()
model.ravel()
y=y.ravel() #Converting to 1D array
# print(model)
# av5 = 

# av10 = 

# av25 =

# av50 =
yensemble= [0]*4
ytakeda=[0]*4
for i in range(0,80):
    if(trees[i] == 5  and model[i] == 'takeda  '):
        ytakeda[0] += y[i] 
        # print(y[i])
    if(trees[i] == 5 and model[i] == "ensemble  " ):
        yensemble[0] += y[i]
    if(trees[i] == 10 and model[i] == "takeda  " ):
        ytakeda[1] += y[i] 
    if(trees[i] == 10 and model[i] == "ensemble  " ):
        yensemble[1] += y[i]
    if(trees[i] == 25 and model[i] == "takeda  " ):
        ytakeda[2] += y[i] 
    if(trees[i] == 25 and model[i] == "ensemble  " ):
        yensemble[2] += y[i]
    if(trees[i] == 50 and model[i] == "takeda  "):
        ytakeda[3] += y[i] 
    if(trees[i] == 50 and model[i] == "ensemble  "):
        yensemble[3] += y[i]
    
print(ytakeda)
print(yensemble)
ytakeda = np.array(ytakeda)
yensemble = np.array(yensemble)
ytakeda = ytakeda/10
yensemble = yensemble/10
# ytakeda= ytakeda/10
# print(ytakeda)

# yensemble=[80.34,79.48,82.05]
# ytakeda=[77.77,76.92]
xx=[5,10,25,50]
GPU1=plt.plot(xx,yensemble,label='ensemble approach')
# plt.plot(xx,ytakeda,label='takeda approach')
# plt.scatter(xx,yta)
# plt.scatter(xx,yy1)
plt.legend()
plt.title("Metrics Analysis")
plt.ylabel("Time(s)")
plt.xlabel("No of trees")
plt.show()
# Importing necessary libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report, confusion_matrix, r2_score
import matplotlib.pyplot as plt
from fast_ml.model_development import train_valid_test_split
import numpy as np
import tensorflow as tf


target = "Vote(Yes/No)"  # Target variable is whether the target agent voted for us or not

train_df = pd.read_csv('takeda training data.csv')


feature_cols = ['negative talks','positive talks','Negative length'] 

x = train_df.drop(target, axis=1).values
y = train_df[[target]].values
y=y.ravel() #Converting to 1D array


#takeda model
X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(train_df, target = 'Vote(Yes/No)', 
                                                                            train_size=0.6, valid_size=0.2, test_size=0.2)

# # print(X_train.shape)
# print(y_train.shape)
# # print(X_valid.shape)
# print(y_valid.shape)
# # print(X_test.shape), 
# print(y_test.shape)

from sklearn.ensemble import RandomForestClassifier #For random forest
takeda_forest_model=RandomForestClassifier(n_estimators=100)


takeda_forest_model.fit(X_train,y_train)
y_predtrain = takeda_forest_model.predict(X_train)
print()
print("Training accuracy of random forest on takeda dataset :",metrics.accuracy_score(y_train, y_predtrain))
print()
# Also calculating the test accuracy for this model to compare it at last
y_pred=takeda_forest_model.predict(X_test)

#*******************************************************************************************************************************
#wasabi model

target = "Vote(Yes/No)"  # Target variable is whether the target agent voted for us or not

train_df = pd.read_csv('wasabi training data.csv')
feature_cols = ['negative talks','positive talks','Negative length'] 

x2 = train_df.drop(target, axis=1).values
y2 = train_df[[target]].values
y2=y2.ravel() #Converting to 1D array


X2_train, y2_train, X2_valid, y2_valid, X2_test, y2_test = train_valid_test_split(train_df, target = 'Vote(Yes/No)', 
                                                                            train_size=0.6, valid_size=0.2, test_size=0.2)

# print(X2_train.shape), print(y2_train.shape)
# print(X2_valid.shape), print(y2_valid.shape)
# print(X2_test.shape), print(y2_test.shape)

wasabi_forest_model=RandomForestClassifier(n_estimators=100)


wasabi_forest_model.fit(X2_train,y2_train)
y2_predtrain = wasabi_forest_model.predict(X2_train)

print("Training accuracy of random forest on wasabi dataset :",metrics.accuracy_score(y2_train, y2_predtrain))
print()
#*******************************************************************************************************************************


# reporter model  

target = "Vote(Yes/No)"  # Target variable is whether the target agent voted for us or not

train_df = pd.read_csv('reporter training data.csv')

# feature_cols = ['negative talks','positive talks','Negative length'] 

# x3 = train_df.drop(target, axis=1).values

# y3 = train_df[[target]].values

# y3=y3.ravel() #Converting to 1D array


X3_train, y3_train, X3_valid, y3_valid, X3_test, y3_test = train_valid_test_split(train_df, target = 'Vote(Yes/No)', train_size=0.6, valid_size=0.2, test_size=0.2)

# print(X3_train.shape), print(y3_train.shape)
# print(X3_valid.shape), print(y3_valid.shape)
# print(X3_test.shape), print(y3_test.shape)


reporter_forest_model=RandomForestClassifier(n_estimators=100)


reporter_forest_model.fit(X3_train,y3_train)
y3_predtrain = reporter_forest_model.predict(X3_train)

print("Training accuracy of random forest on reporter dataset :",metrics.accuracy_score(y3_train, y3_predtrain))
print()


#***************************************************************************************************************************************


# Sample player model

target = "Vote(Yes/No)"  # Target variable is whether the target agent voted for us or not

train_df = pd.read_csv('sample training data.csv')
feature_cols = ['negative talks','positive talks','Negative length'] 

x4 = train_df.drop(target, axis=1).values
y4 = train_df[[target]].values
y4=y4.ravel() #Converting to 1D array


X4_train, y4_train, X4_valid, y4_valid, X4_test, y4_test = train_valid_test_split(train_df, target = 'Vote(Yes/No)', 
                                                                            train_size=0.6, valid_size=0.2, test_size=0.2)

# print(X4_train.shape), print(y4_train.shape)
# print(X4_valid.shape), print(y4_valid.shape)
# print(X4_test.shape), print(y4_test.shape)


sample_forest_model=RandomForestClassifier(n_estimators=100)


sample_forest_model.fit(X4_train,y4_train)
y4_predtrain = sample_forest_model.predict(X4_train)

print("Training accuracy of random forest on sample agent dataset :",metrics.accuracy_score(y4_train, y4_predtrain))
print()

#***************************************************************************************************************************************

# Applying weighted ensemble learning
model_weights = [1]*4


takeda = takeda_forest_model.predict(X_valid)
wasabi = wasabi_forest_model.predict(X_valid)
reporter = reporter_forest_model.predict(X_valid)
sample = sample_forest_model.predict(X_valid)


yvalid = np.array(y_valid)
# Applying weighted ensemble learning
model_weights = [1]*4


for i in range(len(X_valid)):
    
    ans = [0,0,0,0]
    if(yvalid[i] == takeda[i]):
        ans[0] = 1 
    if(yvalid[i] == wasabi[i]):
        ans[1] = 1 
    if(yvalid[i] == reporter[i]):
        ans[2] = 1 
    if(yvalid[i] == sample[i]):
        ans[3] = 1 
    
    if(ans[0]==1):
        model_weights[0] += ans.count(0)/len(ans)

    if(ans[1]==1):
        model_weights[1]+= ans.count(0)/len(ans)

    if(ans[2]==1):
        model_weights[2]+= ans.count(0)/len(ans)

    if(ans[3]==1):
        model_weights[3]+= ans.count(0)/len(ans)

    # Remove comment to see how weights are changing    
    # print(ans)
    # print(model_weights)
    # print("************")

print("The weights of the models are:",model_weights )
print()
# Prediction on test dataset
takeda1 = takeda_forest_model.predict(X_test)
wasabi1 = wasabi_forest_model.predict(X_test)
reporter1 = reporter_forest_model.predict(X_test)
sample1 = sample_forest_model.predict(X_test)

output = [0]*len(X_test)
checker = [0,0,0,0]
ones =0
zeroes =0
for i in range(len(X_test)):
    checker = [0,0,0,0]
    ones = 0
    zeroes = 0
    if(takeda1[i]==1):
        checker[0] = 1

    if(wasabi1[i]==1):
        checker[1] = 1

    if(reporter1[i]==1):
        checker[2] = 1

    if(sample1[i]==1):
        checker[3] = 1

    for j in range(len(checker)):
        if(checker[j]==1):
            ones += model_weights[j]

        else:
            zeroes+= model_weights[j]
    # print("zeroes = ",zeroes)
    # print("ones = ",ones)
    if(ones>zeroes):
        output[i] = 1   
    else:
        output[i] = 0
    
    # print(output[i])


# print(output)
print("The test accuracy of weighted ensemble voting method is : ",metrics.accuracy_score(y_test, output) )
print()
print("Test accuracy of traditional random forest using takeda model is :",metrics.accuracy_score(y_test, y_pred))
print()
# Plotting the confusion matrix

# cm = confusion_matrix(y_test, output)

# fig, ax = plt.subplots(figsize=(8, 8))
# ax.imshow(cm)
# ax.grid(False)
# ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
# ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
# ax.set_ylim(1.5, -0.5)
# for i in range(2):
#     for j in range(2):
#         ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
# plt.show()





# Plotting the confusion matrix

# cm = confusion_matrix(y_test, y_pred)

# fig, ax = plt.subplots(figsize=(8, 8))
# ax.imshow(cm)
# ax.grid(False)
# ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
# ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
# ax.set_ylim(1.5, -0.5)
# for i in range(2):
#     for j in range(2):
#         ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
# plt.show()
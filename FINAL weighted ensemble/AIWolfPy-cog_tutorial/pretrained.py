# Importing necessary libraries
# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
# from sklearn.model_selection import train_test_split # Import train_test_split function
# from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
# from sklearn.metrics import classification_report, confusion_matrix, r2_score
# import matplotlib.pyplot as plt
# from fast_ml.model_development import train_valid_test_split
# import numpy as np
# import tensorflow as tf


import pickle #Importing picke module to save pretrained models

# import time # Importing to calculate time


# ntrees = 5 # No of trees

# import logging, json  # to generate log file

# # logging.basicConfig(filename="metrics data"+".log",level=logging.DEBUG,format='')
# # logging.debug("model,trees,accuracy,time(s)")






# target = "Vote(Yes/No)"  # Target variable is whether the target agent voted for us or not

# train_df = pd.read_csv('takeda training data.csv')


# feature_cols = ['negative talks','positive talks','Negative length'] 

# x = train_df.drop(target, axis=1).values
# y = train_df[[target]].values
# y=y.ravel() #Converting to 1D array


# # #takeda model

# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3) 

# X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(train_df, target = 'Vote(Yes/No)', 
#                                                                             train_size=0.6, valid_size=0.2, test_size=0.2)

# # # print(X_train.shape)
# # print(y_train.shape)
# # # print(X_valid.shape)
# # print(y_valid.shape)
# # # print(X_test.shape), 
# # print(y_test.shape)

# from sklearn.ensemble import RandomForestClassifier #For random forest
# takeda_forest_model=RandomForestClassifier(n_estimators= ntrees)


# takeda_forest_model.fit(X_train,y_train)
# y_predtrain = takeda_forest_model.predict(X_train)
# # print()
# # print("Training accuracy of random forest on takeda dataset :",metrics.accuracy_score(y_train, y_predtrain))
# # print()
# # Also calculating the test accuracy for this model to compare it at last
# # y_pred=takeda_forest_model.predict(X_test)

# #*******************************************************************************************************************************
# #wasabi model

# target = "Vote(Yes/No)"  # Target variable is whether the target agent voted for us or not

# train_df = pd.read_csv('wasabi training data.csv')
# feature_cols = ['negative talks','positive talks','Negative length'] 

# x2 = train_df.drop(target, axis=1).values
# y2 = train_df[[target]].values
# y2=y2.ravel() #Converting to 1D array


# X2_train, y2_train, X2_valid, y2_valid, X2_test, y2_test = train_valid_test_split(train_df, target = 'Vote(Yes/No)', 
#                                                                             train_size=0.6, valid_size=0.2, test_size=0.2)

# X2_train, X2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.3) 

# # print(X2_train.shape), print(y2_train.shape)
# # print(X2_valid.shape), print(y2_valid.shape)
# # print(X2_test.shape), print(y2_test.shape)

# wasabi_forest_model=RandomForestClassifier(n_estimators= ntrees)


# wasabi_forest_model.fit(X2_train,y2_train)
# y2_predtrain = wasabi_forest_model.predict(X2_train)

# # print("Training accuracy of random forest on wasabi dataset :",metrics.accuracy_score(y2_train, y2_predtrain))
# # print()
# #*******************************************************************************************************************************


# # sample agent model  

# target = "Vote(Yes/No)"  # Target variable is whether the target agent voted for us or not

# train_df = pd.read_csv('sample training data.csv')

# feature_cols = ['negative talks','positive talks','Negative length'] 

# x3 = train_df.drop(target, axis=1).values

# y3 = train_df[[target]].values

# y3=y3.ravel() #Converting to 1D array


# X3_train, y3_train, X3_valid, y3_valid, X3_test, y3_test = train_valid_test_split(train_df, target = 'Vote(Yes/No)', train_size=0.6, valid_size=0.2, test_size=0.2)

# X3_train, X3_test, y3_train, y3_test = train_test_split(x3, y3, test_size=0.3) 

# # print(X3_train.shape), print(y3_train.shape)
# # print(X3_valid.shape), print(y3_valid.shape)
# # print(X3_test.shape), print(y3_test.shape)


# sample_forest_model=RandomForestClassifier(n_estimators= ntrees)


# sample_forest_model.fit(X3_train,y3_train)
# y3_predtrain = sample_forest_model.predict(X3_train)

# # print("Training accuracy of random forest on sample dataset :",metrics.accuracy_score(y3_train, y3_predtrain))
# # print()


# #***************************************************************************************************************************************


# # viking agent model

# target = "Vote(Yes/No)"  # Target variable is whether the target agent voted for us or not

# train_df = pd.read_csv('viking training data.csv')
# feature_cols = ['negative talks','positive talks','Negative length'] 

# x4 = train_df.drop(target, axis=1).values
# y4 = train_df[[target]].values
# y4=y4.ravel() #Converting to 1D array


# X4_train, y4_train, X4_valid, y4_valid, X4_test, y4_test = train_valid_test_split(train_df, target = 'Vote(Yes/No)', 
#                                                                             train_size=0.6, valid_size=0.2, test_size=0.2)

# X4_train, X4_test, y4_train, y4_test = train_test_split(x4, y4, test_size=0.3) 

# # print(X4_train.shape), print(y4_train.shape)
# # print(X4_valid.shape), print(y4_valid.shape)
# # print(X4_test.shape), print(y4_test.shape)


# viking_forest_model=RandomForestClassifier(n_estimators= ntrees)


# viking_forest_model.fit(X4_train,y4_train)
# y4_predtrain = viking_forest_model.predict(X4_train)

# # print("Training accuracy of random forest on viking agent dataset :",metrics.accuracy_score(y4_train, y4_predtrain))
# # print()

# #***************************************************************************************************************************************

# # daisyo agent model

# target = "Vote(Yes/No)"  # Target variable is whether the target agent voted for us or not

# train_df = pd.read_csv('daisyo training data.csv')
# feature_cols = ['negative talks','positive talks','Negative length'] 

# x5 = train_df.drop(target, axis=1).values
# y5 = train_df[[target]].values
# y5=y5.ravel() #Converting to 1D array


# X4_train, y4_train, X4_valid, y4_valid, X4_test, y4_test = train_valid_test_split(train_df, target = 'Vote(Yes/No)', 
#                                                                             train_size=0.6, valid_size=0.2, test_size=0.2)

# X5_train, X5_test, y5_train, y5_test = train_test_split(x5, y5, test_size=0.3) 

# # print(X4_train.shape), print(y4_train.shape)
# # print(X4_valid.shape), print(y4_valid.shape)
# # print(X4_test.shape), print(y4_test.shape)


# daisyo_forest_model=RandomForestClassifier(n_estimators= ntrees)


# daisyo_forest_model.fit(X5_train,y5_train)
# y5_predtrain = daisyo_forest_model.predict(X5_train)

# # print("Training accuracy of random forest on daisyo agent dataset :",metrics.accuracy_score(y5_train, y5_predtrain))
# # print()











#save the models

# with open('takeda_model','wb') as f:
#     pickle.dump(takeda_forest_model,f)

# with open('wasabi_model','wb') as f:
#     pickle.dump(wasabi_forest_model,f)

# with open('viking_model','wb') as f:
#     pickle.dump(viking_forest_model,f)

# with open('sample_model','wb') as f:
#     pickle.dump(sample_forest_model,f)

# with open('daisyo_model','wb') as f:
#     pickle.dump(daisyo_forest_model,f)


#use the models

# with open('takeda_model','rb') as f:
#     takeda_forest_model = pickle.load(f)

# with open('wasabi_model','rb') as f:
#     wasabi_forest_model = pickle.load(f)
# with open('viking_model','rb') as f:
#     viking_forest_model = pickle.load(f)
# with open('sample_model','rb') as f:
#     sample_forest_model = pickle.load(f)
# with open('daisyo_model','rb') as f:
#     daisyo_forest_model = pickle.load(f)











#***************************************************************************************************************************************
# Applying weighted ensemble learning
# model_weights = [1]*5


# takeda = takeda_forest_model.predict(X_test)
# wasabi = wasabi_forest_model.predict(X_test)
# sample = sample_forest_model.predict(X_test)
# viking = viking_forest_model.predict(X_test)
# daisyo = daisyo_forest_model.predict(X_test)

# # yvalid = np.array(y_valid)
# # Applying weighted ensemble learning
# model_weights = [1]*5


# for i in range(len(X_test)):

#     ans = [0,0,0,0,0]
#     if(y_test[i] == takeda[i]):
#         ans[0] = 1 
#     if(y_test[i] == wasabi[i]):
#         ans[1] = 1 
#     if(y_test[i] == sample[i]):
#         ans[2] = 1 
#     if(y_test[i] == viking[i]):
#         ans[3] = 1 
#     if(y_test[i] == daisyo[i]):
#         ans[4] = 1
    
#     if(ans[0]==1):
#         model_weights[0] += ans.count(0)/len(ans)

#     if(ans[1]==1):
#         model_weights[1]+= ans.count(0)/len(ans)

#     if(ans[2]==1):
#         model_weights[2]+= ans.count(0)/len(ans)

#     if(ans[3]==1):
#         model_weights[3]+= ans.count(0)/len(ans)

#     if(ans[4]==1):
#         model_weights[4]+= ans.count(0)/len(ans)

#     # Remove comment to see how weights are changing    
#     # print(ans)
#     # print(model_weights)
#     # print("************")

# # print("The weights of the models are:",model_weights )
# # print()


# takeda = takeda_forest_model.predict(X2_test)
# wasabi = wasabi_forest_model.predict(X2_test)
# sample = sample_forest_model.predict(X2_test)
# viking = viking_forest_model.predict(X2_test)
# daisyo = daisyo_forest_model.predict(X2_test)

# # print("THe length 1 is =" ,len(daisyo))
# # print("THe length 1 is =" ,len(X2_test))
# for i in range(len(X2_test)):
    
#     ans = [0,0,0,0,0]
#     if(y2_test[i] == takeda[i]):
#         ans[0] = 1 
#     if(y2_test[i] == wasabi[i]):
#         ans[1] = 1 
#     if(y2_test[i] == sample[i]):
#         ans[2] = 1 
#     if(y2_test[i] == viking[i]):
#         ans[3] = 1 
#     if(y2_test[i] == daisyo[i]):
#         ans[4] = 1
    
#     if(ans[0]==1):
#         model_weights[0] += ans.count(0)/len(ans)

#     if(ans[1]==1):
#         model_weights[1]+= ans.count(0)/len(ans)

#     if(ans[2]==1):
#         model_weights[2]+= ans.count(0)/len(ans)

#     if(ans[3]==1):
#         model_weights[3]+= ans.count(0)/len(ans)
    
#     if(ans[4]==1):
#         model_weights[4]+= ans.count(0)/len(ans)


# takeda = takeda_forest_model.predict(X3_test)
# wasabi = wasabi_forest_model.predict(X3_test)
# sample = sample_forest_model.predict(X3_test)
# viking = viking_forest_model.predict(X3_test)
# daisyo = daisyo_forest_model.predict(X3_test)

# # print("The weights of the models are:",model_weights )
# # print()


# for i in range(len(X3_test)):
    
#     ans = [0,0,0,0,0]
#     if(y3_test[i] == takeda[i]):
#         ans[0] = 1 
#     if(y3_test[i] == wasabi[i]):
#         ans[1] = 1 
#     if(y3_test[i] == sample[i]):
#         ans[2] = 1 
#     if(y3_test[i] == viking[i]):
#         ans[3] = 1 
#     if(y3_test[i] == daisyo[i]):
#         ans[4] = 1

#     if(ans[0]==1):
#         model_weights[0] += ans.count(0)/len(ans)

#     if(ans[1]==1):
#         model_weights[1]+= ans.count(0)/len(ans)

#     if(ans[2]==1):
#         model_weights[2]+= ans.count(0)/len(ans)

#     if(ans[3]==1):
#         model_weights[3]+= ans.count(0)/len(ans)

#     if(ans[4]==1):
#         model_weights[4]+= ans.count(0)/len(ans)

# # print("The weights of the models are:",model_weights )
# # print()

# takeda = takeda_forest_model.predict(X4_test)
# wasabi = wasabi_forest_model.predict(X4_test)
# sample = sample_forest_model.predict(X4_test)
# viking = viking_forest_model.predict(X4_test)
# daisyo = daisyo_forest_model.predict(X4_test)

# for i in range(len(X4_test)):
    
#     ans = [0,0,0,0,0]
#     if(y4_test[i] == takeda[i]):
#         ans[0] = 1 
#     if(y4_test[i] == wasabi[i]):
#         ans[1] = 1 
#     if(y4_test[i] == sample[i]):
#         ans[2] = 1 
#     if(y4_test[i] == viking[i]):
#         ans[3] = 1 
#     if(y4_test[i] == daisyo[i]):
#         ans[4] = 1


#     if(ans[0]==1):
#         model_weights[0] += ans.count(0)/len(ans)

#     if(ans[1]==1):
#         model_weights[1]+= ans.count(0)/len(ans)

#     if(ans[2]==1):
#         model_weights[2]+= ans.count(0)/len(ans)

#     if(ans[3]==1):
#         model_weights[3]+= ans.count(0)/len(ans)

#     if(ans[4]==1):
#         model_weights[4]+= ans.count(0)/len(ans)

# # print("The final weights of the models are:",model_weights )
# # print()



# takeda = takeda_forest_model.predict(X5_test)
# wasabi = wasabi_forest_model.predict(X5_test)
# sample = sample_forest_model.predict(X5_test)
# viking = viking_forest_model.predict(X5_test)
# daisyo = daisyo_forest_model.predict(X5_test)

# for i in range(len(X5_test)):
    
#     ans = [0,0,0,0,0]
#     if(y5_test[i] == takeda[i]):
#         ans[0] = 1 
#     if(y5_test[i] == wasabi[i]):
#         ans[1] = 1 
#     if(y5_test[i] == sample[i]):
#         ans[2] = 1 
#     if(y5_test[i] == viking[i]):
#         ans[3] = 1 
#     if(y5_test[i] == daisyo[i]):
#         ans[4] = 1


#     if(ans[0]==1):
#         model_weights[0] += ans.count(0)/len(ans)

#     if(ans[1]==1):
#         model_weights[1]+= ans.count(0)/len(ans)

#     if(ans[2]==1):
#         model_weights[2]+= ans.count(0)/len(ans)

#     if(ans[3]==1):
#         model_weights[3]+= ans.count(0)/len(ans)

#     if(ans[4]==1):
#         model_weights[4]+= ans.count(0)/len(ans)

# print("The final weights of the models are:",model_weights )
# print()





def predict_vote(N_talks,P_talks,N_length):


    model_weights = [25.399999999999967, 22.2, 24.799999999999976, 26.399999999999974, 15.399999999999993]


    with open('takeda_model','rb') as f:
        takeda_forest_model = pickle.load(f)

    with open('wasabi_model','rb') as f:
        wasabi_forest_model = pickle.load(f)
    with open('viking_model','rb') as f:
        viking_forest_model = pickle.load(f)
    with open('sample_model','rb') as f:
        sample_forest_model = pickle.load(f)
    with open('daisyo_model','rb') as f:
        daisyo_forest_model = pickle.load(f)





    # train_df = pd.read_csv('test data.csv')


    # feature_cols = ['negative talks','positive talks','Negative length'] 

    # x5 = train_df.drop(target, axis=1).values
    # y5 = train_df[[target]].values
    # y5=y5.ravel() #Converting to 1D array


    #ensemble model###############################################################################################################################################

    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.0) 


    x5 = [[N_talks,P_talks,N_length]]


    # x5 = [[0,10,0]]


    takeda1 = takeda_forest_model.predict(x5)
    wasabi1 = wasabi_forest_model.predict(x5)
    sample1 = sample_forest_model.predict(x5)
    viking1 = viking_forest_model.predict(x5)
    daisyo1 = daisyo_forest_model.predict(x5)

    # print("takeda 1 =",takeda1[0])
    output = [0]*len(x5)
    output2 = [0]*len(x5)
    checker = [0,0,0,0,0]
    ones =0
    zeroes =0

    checker = [0,0,0,0,0]
    ones = 0
    zeroes = 0
    if(takeda1[0]==1):
        checker[0] = 1

    if(wasabi1[0]==1):
        checker[1] = 1

    if(sample1[0]==1):
        checker[2] = 1

    if(viking1[0]==1):
        checker[3] = 1

    if(daisyo1[0]==1):
        checker[4] = 1

    # if(checker.count(1)>checker.count(0)):
    #     output2[i] = 1
    # else:
    #     output2[i] = 0

    for j in range(len(checker)):
        if(checker[j]==1):
            ones += model_weights[j]

        else:
            zeroes+= model_weights[j]
    # print("zeroes = ",zeroes)
    # print("ones = ",ones)
    if(ones>zeroes):
        output[0] = 1   
    else:
        output[0] = 0

    # print("The output is =",output[0])


    return output[0]



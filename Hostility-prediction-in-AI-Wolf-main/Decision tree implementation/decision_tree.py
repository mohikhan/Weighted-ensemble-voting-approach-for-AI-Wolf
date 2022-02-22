# Decision tree implementation to predict whether the agent will
# vote for me or not in the ai wolf game

#importing necessary libraries

import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

target = "Vote(Yes/No)"  # Target variable is whether the target agent voted for us or not



train_df= pd.read_csv("training data.csv")

feature_cols = ['negative talks','positive talks','Negative length'] 

x = train_df.drop(target, axis=1).values
y = train_df[[target]].values
y=y.ravel() #Converting to 1D array

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1) # 70% training and 30% test

tree_model = DecisionTreeClassifier()

# Train Decision Tree Classifer
tree_model = tree_model.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = tree_model.predict(X_test)




# Training data statistics*************************************************************************

y_predtrain = tree_model.predict(X_train)

print("Training accuracy of decision tree on takeda dataset :",metrics.accuracy_score(y_train, y_predtrain))

print("The confusion matrix of training dataset: ")
print(confusion_matrix(y_train, y_predtrain))


# Plotting the confusion matrix
cm = confusion_matrix(y_train, y_predtrain)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()


# Test data statistics******************************************************************


y_pred=tree_model.predict(X_test)

print("Test accuracy of decision on takeda dataset :",metrics.accuracy_score(y_test, y_pred))

print("The confusion matrix of test dataset: ")
print(confusion_matrix(y_test, y_pred))

# Plotting the confusion matrix

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()


#Printing the decision tree


from sklearn.tree import export_graphviz
# from sklearn.externals.six import StringIO  
from six import StringIO

from IPython.display import Image  
import pydotplus
dot_data = StringIO()
export_graphviz(tree_model, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('DT.png')
Image(graph.create_png())


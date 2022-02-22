#Importing necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

target = "Vote(Yes/No)"  # Target variable is whether the target agent voted for us or not

train_df= pd.read_csv("training data.csv")

x = train_df.drop(target, axis=1).values
y = train_df[[target]].values
y=y.ravel() #Converting to 1D array


# defining logistic regression model
model = LogisticRegression(solver='liblinear', random_state=0)

#Fitting the model on the data
model.fit(x, y)

# print( model.predict(x))
print("The accuracy  of the model is: ")
print(model.score(x, y) *100)


#Printing additional metrics (confusion matrix)
print(confusion_matrix(y, model.predict(x)))


cm = confusion_matrix(y, model.predict(x))

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
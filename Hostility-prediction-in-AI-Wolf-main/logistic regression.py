import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

target = "Vote(Yes/No)"  # Target variable is whether the target agent voted for us or not

train_df= pd.read_csv("training data.csv")

x = train_df.drop(target, axis=1).values
y = train_df[[target]].values
y=y.ravel()
model = LogisticRegression(solver='liblinear', random_state=0)

model.fit(x, y)

model = LogisticRegression(solver='liblinear', random_state=0).fit(x, y)
# print( model.predict(x))
print(model.score(x, y))

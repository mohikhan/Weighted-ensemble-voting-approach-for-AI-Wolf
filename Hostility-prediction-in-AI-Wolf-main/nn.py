
#Importing necessary libraries
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


from sklearn.preprocessing import MinMaxScaler

target = 'vote'  # Target variable is whether the target agent voted for us or not

train_df= pd.read_csv("FIle path dataset")

scaler = MinMaxScaler(feature_range=(0, 1))
# Scale both the training inputs and outputs
scaled_train = scaler.fit_transform(train_df)
#scaled_test = scaler.transform(test_df)

print("Note: median values were scaled by multiplying by {:.10f} and adding {:.6f}".format(scaler.scale_[13], scaler.min_[13]))

multiplied_by = scaler.scale_[2]
added = scaler.min_[2]

scaled_train_df = pd.DataFrame(scaled_train, columns=train_df.columns.values)


#Neural network model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')


tr = ["negative talks","positive talks"]

X = scaled_train_df.drop(tr, axis=1).values
Y = scaled_train_df[[target]].values

model.fit(
    X,
    Y,
    epochs=50,
    shuffle=True,
    verbose=2
)

test_error_rate = model.evaluate(X, Y, verbose=0)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))

prediction = model.predict(X)


#Calculate metrics .....



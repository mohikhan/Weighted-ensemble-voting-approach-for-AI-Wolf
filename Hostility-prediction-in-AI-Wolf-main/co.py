#Convert log file into csv


# importing panda library
import pandas as pd
  
# readinag given csv file
# and creating dataframe
dataframe1 = pd.read_csv("training data.log")
  
# storing this dataframe in a csv file
dataframe1.to_csv('training data.csv', index = None)
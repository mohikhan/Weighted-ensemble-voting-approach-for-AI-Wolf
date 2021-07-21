import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train_df = pd.read_csv('5 player metrics data.csv')

train_df2 = pd.read_csv('15 player metrics data.csv')

target1 = "accuracy"
y = train_df[[target1]].values
y= y.ravel()
y2 = train_df2[[target1]].values
y2 = y2.ravel()

x = [1,2,3,4,5,6,7,8,9,10]

# plt.plot(x,y)

#Plot accuracy

plt.plot(x,y,label='5 player ens. approach')
plt.plot(x,y2,label='15 player ens. approach')
# plt.scatter(xx,yta)
# plt.scatter(xx,yy1)
plt.legend()
plt.title("Metrics Analysis")
plt.ylabel("Accuracy%")
plt.xlabel("Iteration")
plt.show()

#___________________________________________________________________________________________________________________
#plot f1 score
target1 = "f1 score"
y = train_df[[target1]].values
y= y.ravel()
y2 = train_df2[[target1]].values
y2 = y2.ravel()

x = [1,2,3,4,5,6,7,8,9,10]

# plt.plot(x,y)

plt.plot(x,y,label='5 player ens. approach')
plt.plot(x,y2,label='15 player ens. approach')
# plt.scatter(xx,yta)
# plt.scatter(xx,yy1)
plt.legend()
plt.title("Metrics Analysis")
plt.ylabel("fscore")
plt.xlabel("Iteration")
plt.show()

#___________________________________________________________________________________________________________________
#Plot false positives

target1 = "false positives"
y = train_df[[target1]].values
y= y.ravel()
y2 = train_df2[[target1]].values
y2 = y2.ravel()

x = [1,2,3,4,5,6,7,8,9,10]

# plt.plot(x,y)

plt.plot(x,y,label='5 player ens. approach')
plt.plot(x,y2,label='15 player ens. approach')
# plt.scatter(xx,yta)
# plt.scatter(xx,yy1)
plt.legend()
plt.title("Metrics Analysis")
plt.ylabel("false positives")
plt.xlabel("Iteration")
plt.show()

#___________________________________________________________________________________________________________________
#plot false negatives

target1 = "false negatives"
y = train_df[[target1]].values
y= y.ravel()
y2 = train_df2[[target1]].values
y2 = y2.ravel()

x = [1,2,3,4,5,6,7,8,9,10]

# plt.plot(x,y)

plt.plot(x,y,label='5 player ens. approach')
plt.plot(x,y2,label='15 player ens. approach')
# plt.scatter(xx,yta)
# plt.scatter(xx,yy1)
plt.legend()
plt.title("Metrics Analysis")
plt.ylabel("false negatives")
plt.xlabel("Iteration")
plt.show()
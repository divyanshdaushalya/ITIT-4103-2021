import numpy as np
import pandas as pd
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import KFold

input=np.random.random((50,1))
noiseless=np.sin(1+np.square(input))
noise=np.random.normal(0,0.032,size=(50,1))
noisefull=noiseless+noise

Ytrain=noisefull[:40]
Ytest=noisefull[40:]
Xtrain=input[:40]
Xtest=input[40:]

b = Ytrain.reshape(Ytrain.shape[0],1)
A = np.concatenate((Xtrain, np.ones((Xtrain.shape[0], 1))), axis=1)
z = np.dot(np.dot(inv(np.dot(A.T,A)),A.T),b)

y1 = z[0] * Xtrain + z[1]
y2 = z[0] * Xtest + z[1]

plt.subplot(1,2,1)
plt.title('Training set')
plt.scatter(Xtrain, y1,label='model output',c='r')
plt.scatter(Xtrain, Ytrain,label='actual output',c='g')
plt.legend()

plt.subplot(1,2,2)
plt.title('Testing set')
plt.scatter(Xtest, y2,label='model output',c='r')
plt.scatter(Xtest, Ytest,label='actual output',c='g')
plt.legend()

MSE = np.square(np.subtract(Ytrain, y1)).mean()
RMSE = math.sqrt(MSE)
MSE1 = np.square(np.subtract(Ytest, y2)).mean()
RMSE1 = math.sqrt(MSE1)
print('train rmse - ',RMSE,'test rmse -',RMSE)

plt.show()

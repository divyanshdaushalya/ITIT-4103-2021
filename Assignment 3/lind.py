import numpy as np
import pandas as pd
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import KFold

input=np.random.random((50,1))
noiseless=np.sin(1+np.square(input))
noise=np.random.normal(0,0.032,size=(50,1))

noisefull=noiseless+noise

b = noisefull.reshape(noisefull.shape[0],1)
A = np.concatenate((input, np.ones((input.shape[0], 1))), axis=1)
z = inv(A.T @ A) @ A.T @ b

y1 = z[0] * input + z[1]
print(z[0],z[1])

plt.scatter(input, y1)
plt.scatter(input, noisefull)
plt.show()

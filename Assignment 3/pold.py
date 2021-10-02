
import numpy as np
from matplotlib import pyplot as plt
import math

#####################################
############ DIRECT METHOD ##########
#####################################
print("Direct Method")


def getPoweredX(deg, X_t):
    X_pow = X_t.copy()
    for i in range(2, deg + 1):
        powmat = (X_t[:, 1] ** i).reshape(X_pow.shape[0], 1)
        X_pow = np.append(X_pow, powmat, axis=1)
    return X_pow


def calculateEmpricalRisk(yh):
    error = sum((Y - yh) ** 2)
    return error / len(Y)

input=np.random.random((50,1))
noiseless=np.sin(1+np.square(input))
noise=np.random.normal(0,0.032,size=(50,1))
noisefull=noiseless+noise

Ytrain=noisefull[:40]
Ytest=noisefull[40:]
Xtrain=input[:40]
Xtest=input[40:]

X_train = np.append(np.ones((Xtrain.shape[0], 1)), Xtrain, axis=1)
Y = Ytrain
X_test=np.append(np.ones((Xtest.shape[0], 1)), Xtest, axis=1)

fig, ((ax1, ax2), (ax3, ax4),(ax5,ax6),(ax7,ax8)) = plt.subplots(4, 2)

# For degree 1 polynomial
deg1 = 1
X1 = getPoweredX(deg1, X_train)
first_part = np.linalg.inv(np.dot(X1.transpose(), X1))
second_part = np.dot(first_part, X1.transpose())
W1 = np.dot(second_part, Y)

XT1=getPoweredX(deg1, X_test)
YTrainPredict=np.dot(X1, W1)
YTestPredict=np.dot(XT1, W1)

ax1.scatter(Xtrain, Ytrain)
ax1.scatter(X1[:,1], YTrainPredict)
ax1.set_title("Degree=1 Train")
ax2.scatter(Xtest, Ytest)
ax2.scatter(XT1[:,1], YTestPredict)
ax2.set_title("Degree=1 Test")

MSE = np.square(np.subtract(Ytrain,YTrainPredict )).mean()
RMSE = math.sqrt(MSE)
MSE1 = np.square(np.subtract(Ytest, YTestPredict)).mean()
RMSE1 = math.sqrt(MSE1)
print('Degree 1 train rmse - ',RMSE,'test rmse -',RMSE1)


# For degree 2 polynomial
deg2 = 2
X2 = getPoweredX(deg2, X_train)

first_part = np.linalg.pinv(np.dot(X2.transpose(), X2))
second_part = np.dot(first_part, X2.transpose())
W2 = np.dot(second_part, Y)

XT1=getPoweredX(deg1, X_test)

XT2=getPoweredX(deg2, X_test)

YTrainPredict=np.dot(X2, W2)
YTestPredict=np.dot(XT2, W2)

ax3.scatter(Xtrain, Ytrain)
ax3.scatter(X2[:,1], YTrainPredict)
ax3.set_title("Degree=2 Train")
ax4.scatter(Xtest, Ytest)
ax4.scatter(XT2[:,1], YTestPredict)
ax4.set_title("Degree=2 Test")


MSE = np.square(np.subtract(Ytrain,YTrainPredict )).mean()
RMSE = math.sqrt(MSE)
MSE1 = np.square(np.subtract(Ytest, YTestPredict)).mean()
RMSE1 = math.sqrt(MSE1)
print('Degree 2 train rmse - ',RMSE,'test rmse -',RMSE1)




# For degree 3 polynomial
deg3 = 3
X3 = getPoweredX(deg3, X_train)

first_part = np.linalg.pinv(np.dot(X3.transpose(), X3))
second_part = np.dot(first_part, X3.transpose())
W3 = np.dot(second_part, Y)

XT3=getPoweredX(deg3, X_test)

YTrainPredict=np.dot(X3, W3)
YTestPredict=np.dot(XT3, W3)

ax5.scatter(Xtrain, Ytrain)
ax5.scatter(X3[:,1], YTrainPredict)
ax5.set_title("Degree=3 Train")
ax6.scatter(Xtest, Ytest)
ax6.scatter(XT3[:,1], YTestPredict)
ax6.set_title("Degree=3 Test")

MSE = np.square(np.subtract(Ytrain,YTrainPredict )).mean()
RMSE = math.sqrt(MSE)
MSE1 = np.square(np.subtract(Ytest, YTestPredict)).mean()
RMSE1 = math.sqrt(MSE1)
print('Degree 3 train rmse - ',RMSE,'test rmse -',RMSE1)


# For degree 4 polynomial
deg4 = 4
X4 = getPoweredX(deg4, X_train)

first_part = np.linalg.pinv(np.dot(X4.transpose(), X4))
second_part = np.dot(first_part, X4.transpose())
W4 = np.dot(second_part, Y)

XT4=getPoweredX(deg4, X_test)
YTrainPredict=np.dot(X4, W4)
YTestPredict=np.dot(XT4, W4)

ax7.scatter(Xtrain, Ytrain)
ax7.scatter(X4[:,1], YTrainPredict)
ax7.set_title("Degree=4 Train")
ax8.scatter(Xtest, Ytest)
ax8.scatter(XT4[:,1], YTestPredict)
ax8.set_title("Degree=4 Test")

MSE = np.square(np.subtract(Ytrain,YTrainPredict )).mean()
RMSE = math.sqrt(MSE)
MSE1 = np.square(np.subtract(Ytest, YTestPredict)).mean()
RMSE1 = math.sqrt(MSE1)
print('Degree 4 train rmse - ',RMSE,'test rmse -',RMSE1)


plt.show()

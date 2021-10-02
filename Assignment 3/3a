
import numpy as np
from matplotlib import pyplot as plt

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

X = input
X_train = np.append(np.ones((X.shape[0], 1)), X, axis=1)
Y = noiseless

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

# For degree 1 polynomial
deg1 = 1
X1 = getPoweredX(deg1, X_train)

first_part = np.linalg.inv(np.dot(X1.transpose(), X1))
second_part = np.dot(first_part, X1.transpose())
W1 = np.dot(second_part, Y)

ax1.scatter(X, Y)
ax1.scatter(X1[:,1], np.dot(X1, W1))
ax1.set_title("Degree=1")
print("Empirical risk for degree 1 model: ", calculateEmpricalRisk(np.dot(X1, W1)))

# For degree 2 polynomial
deg2 = 2
X2 = getPoweredX(deg2, X_train)

first_part = np.linalg.pinv(np.dot(X2.transpose(), X2))
second_part = np.dot(first_part, X2.transpose())
W2 = np.dot(second_part, Y)

ax2.scatter(X, Y)
ax2.scatter(X2[:,1], np.dot(X2, W2))
ax2.set_title("Degree=2")
print("Empirical risk for degree 2 model: ", calculateEmpricalRisk(np.dot(X2, W2)))

# For degree 3 polynomial
deg3 = 3
X3 = getPoweredX(deg3, X_train)

first_part = np.linalg.pinv(np.dot(X3.transpose(), X3))
second_part = np.dot(first_part, X3.transpose())
W3 = np.dot(second_part, Y)

ax3.scatter(X, Y)
ax3.scatter(X3[:,1], np.dot(X3, W3))
ax3.set_title("Degree=3")
print("Empirical risk for degree 3 model: ", calculateEmpricalRisk(np.dot(X3, W3)))

# For degree 4 polynomial
deg4 = 4
X4 = getPoweredX(deg4, X_train)

first_part = np.linalg.pinv(np.dot(X4.transpose(), X4))
second_part = np.dot(first_part, X4.transpose())
W4 = np.dot(second_part, Y)


ax4.scatter(X, Y)

ax4.scatter(X4[:,1],np.dot(X4,W4))
ax4.set_title("Degree=4")
print("Empirical risk for degree 4 model: ", calculateEmpricalRisk(np.dot(X4, W4)))
plt.show()
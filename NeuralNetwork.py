
import numpy as np

xAll = np.array(([2, 9], [1, 5], [3, 6], [5, 10]), dtype=float)  # input
y = np.array(([92], [86], [89]), dtype=float)  # output

# scaling
xAll = xAll/np.amax(xAll, axis=0)
y = y/100

X = np.split(xAll, [3])[0]  # training
xPredicted = np.split(xAll, [3])[1]  # testing


class Neural_Network(object):
    def __init__(self):
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
        self.w1 = np.random.randn(self.inputSize, self.hiddenSize)  # weight1
        self.w2 = np.random.randn(self.hiddenSize, self.outputSize)  # weight2

    def forward(self, X):
        # forawrd propagation
        self.z = np.dot(X, self.w1)  # input(dot)weight1
        self.z2 = self.sigmoid(self.z)  # activation function
        self.z3 = np.dot(self.z2, self.w2)  # hidden(dot)weight2
        o = self.sigmoid(self.z3)  # output layer activation function
        return o

    def sigmoid(self, s):
        # activation function
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self, s):
        return s*(1-s)

    def backward(self, X, y, o):
        self.o_error = y-o  # output error
        self.o_delta = self.o_error*self.sigmoidPrime(o)

        # hidden layer contribution to error
        self.z2_error = self.o_delta.dot(self.w2.T)
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2)

        self.w1 += X.T.dot(self.z2_delta)  # adjustung first weight
        self.w2 += self.z2.T.dot(self.o_delta)  # adjusting second weight

    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)

    def saveWeights(self):
        np.savetxt("w1.txt", self.w1, fmt="%s")
        np.savetxt("w2.txt", self.w2, fmt="%s")

    def predict(self):
        print("Predicted data based on trained weights: ")
        print("Input (scaled): \n" + str(xPredicted))
        print("Output: \n" + str(self.forward(xPredicted)))


NN = Neural_Network()
for i in range(150000):
    print("Input: \n" + str(X))
    print("Actual output: \n" + str(y))
    print("Predicted output: \n" + str(NN.forward(X)))
    print("Loss: \n" + str(np.mean(np.square(y - NN.forward(X)))))
    print("\n")
    NN.train(X, y)

NN.saveWeights()
NN.predict()

"""
Implementation of extreme learning machine, kernel ridge regression, and least squares.
Simon Westberg
Tony Rönnqvist
"""

import numpy as np
from matplotlib import pyplot as plt


def sigmoid(x, w, b):
    return 1 / (1 + np.exp(-1 * (np.dot(x, w) + b)))


def gaussian(x, w, b):
    return np.exp(-1 * (np.dot(x, w) + b) ** 2)


def sin(x, w, b):
    return np.sin(np.dot(x, w) + b)


def relu(x, w, b):
    a = np.dot(x, w) + b
    if a <= 0:
        return 0
    else:
        return a


def softplus(x, w, b):
    X = np.dot(x, w) + b
    return np.log(np.exp(X) + 1)


def MP_inv(H, lam):
    H_T = H.transpose()
    gram = np.dot(H_T, H)
    return np.linalg.solve(gram + lam * np.identity(len(H_T)), H_T)


def Kernel_lin(x, y):
    return np.dot(x, y)


def Kernel_poly(x, y, c, d, gamma):
    return ((1 / gamma) * np.dot(x, y) + c) ** d


def Kernel_RBF(x, y, s):
    n = np.linalg.norm(x - y) ** 2
    a = n / s
    return np.exp(-a)


class ELM:
    """
    X: Array containing input vectors x
    T: Array containing target vectors t
    """

    def __init__(self, X, T):
        self.X = X
        self.T = T
        self.nrNodes = None  # nr of hidden layer nodes
        self.lam = None  # lambda
        self.act = None  # activation function
        self.H = None  # hidden layer matrix
        self.beta = None  # output weights
        self.w = None  # hidden layer weights
        self.b = None  # hidden layer bias

        self.N = len(X)  # nr of samples
        self.D = len(X[0])  # dimension of samples

    def train(self, act, lam, nrNodes):
        self.act = act
        self.lam = lam
        self.nrNodes = nrNodes
        w = []
        b = []

        # Assign random values to the weight vectors and biases for each node
        for i in range(nrNodes):
            w.append(np.random.uniform(-1, 1, self.D))
            b.append(np.random.uniform(-1, 1))

        self.w = w
        self.b = b

        H = np.ndarray(shape=(self.N, nrNodes), dtype=float)

        # Calculate hidden layer matrix
        for i in range(self.N):
            for j in range(nrNodes):
                if act == "sig":
                    H[i][j] = sigmoid(self.X[i], w[j], b[j])
                elif act == "gauss":
                    H[i][j] = gaussian(self.X[i], w[j], b[j])
                elif act == "sin":
                    H[i][j] = sin(self.X[i], w[j], b[j])
                elif act == "relu":
                    H[i][j] = relu(self.X[i], w[j], b[j])
                elif act == "softplus":
                    H[i][j] = softplus(self.X[i], w[j], b[j])
                else:
                    raise ValueError("Invalid activation function")

        self.H = H
        self.beta = np.dot(MP_inv(H, lam), self.T)

    def y(self, x, w, b, beta, act):
        """
        Calculate target vector for input x, using w = weigths, b = biases, beta = output weights,
        with activation function = act.
        """
        t = 0
        for i in range(self.nrNodes):
            if act == "sig":
                t += beta[i] * sigmoid(x, w[i], b[i])
            elif act == "gauss":
                t += beta[i] * gaussian(x, w[i], b[i])
            elif act == "sin":
                t += beta[i] * sin(x, w[i], b[i])
            elif act == "relu":
                t += beta[i] * relu(x, w[i], b[i])
            elif act == "softplus":
                t += beta[i] * softplus(x, w[i], b[i])
            else:
                raise ValueError("Invalid activation function")
        return t

    def __test(self, X_test, T_test, w, b, beta, act):
        correct_1 = 0  # nr of correct predictions for the -1 class
        N_1 = 0  # nr of test inputs of the -1 class
        correct_2 = 0  # nr of correct predictions for the +1 class
        N_2 = 0  # nr of test inputs of the +1 class

        for i in range(len(X_test)):
            target = T_test[i]
            if target[0] == 1:
                N_1 += 1
            elif target[1] == 1:
                N_2 += 1
            else:
                raise ValueError("Something is wrong with the test target vector")

            t = self.y(X_test[i], w, b, beta, act)
            t_argmax = np.argmax(t)
            target = np.argmax(target)
            if t_argmax == target:
                if t_argmax == 0:
                    correct_1 += 1
                elif t_argmax == 1:
                    correct_2 += 1
                else:
                    raise ValueError("Something went wrong in the testing")

        N_test = N_1 + N_2
        assert N_test == len(T_test)
        acc_1 = correct_1 / N_1
        acc_2 = correct_2 / N_2
        accuracy = (correct_1 + correct_2) / N_test
        return accuracy, acc_1, acc_2

    def test(self, X_test, T_test):
        """
        Returns total accuracy, accuracy class 1, accuracy class 2.
        """
        return self.__test(X_test, T_test, self.w, self.b, self.beta, self.act)


class KRR:
    """
    Kernel Ridge Regression
    X: Array containing input vectors x
    T: Array containing target vectors t
    """

    def __init__(self, X, T, kernel, c=0, d=2, s=1, gamma=1):
        self.T = T
        self.X = X
        self.kernel = kernel
        self.weights = None
        self.lam = None  # lambda
        self.N = len(X)  # nr of samples
        self.D = len(X[0])  # dimension of samples
        self.c = c  # hyperparameter polynomial kernel
        self.d = d  # hyperparameter polynomial kernel
        self.s = s  # hyperparameter Gaussian kernel
        self.gamma = gamma  # hyperparameter polynomial kernel

        K = np.ndarray(shape=(self.N, self.N), dtype=float)

        # Calculate kernel matrix
        for i in range(self.N):
            for j in range(self.N):
                if kernel == "lin":
                    K[i][j] = Kernel_lin(X[i], X[j])
                elif kernel == "poly":
                    K[i][j] = Kernel_poly(X[i], X[j], self.c, self.d, self.gamma)
                elif kernel == "rbf":
                    K[i][j] = Kernel_RBF(X[i], X[j], self.s)
                else:
                    raise ValueError("Incorrect kernel")

        self.K = K

    def train(self, lam):
        K_lam = self.K.transpose() + lam * np.identity(self.N)
        f_T = np.linalg.solve(K_lam, self.T)
        self.f = f_T.transpose()

    def y(self, x, f, kernel):
        k = np.ndarray(shape=(self.N, 1))
        for i in range(self.N):
            if kernel == "lin":
                k[i] = Kernel_lin(self.X[i], x)
            elif kernel == "poly":
                k[i] = Kernel_poly(self.X[i], x, self.c, self.d, self.gamma)
            elif kernel == "rbf":
                k[i] = Kernel_RBF(self.X[i], x, self.s)
            else:
                raise ValueError("Incorrect kernel")
        return np.dot(f, k)

    def test(self, X_test, T_test):
        """
        Returns total accuracy, accuracy class 1, accuracy class 2
        """
        correct_1 = 0  # nr of correct predictions of the 1 class
        N_1 = 0  # nr of test inputs of the 1 class
        correct_2 = 0  # nr of correct predictions of the 2 class
        N_2 = 0  # nr of test inputs of the 2 class

        for i in range(len(X_test)):
            target = T_test[i]
            if target[0] == 1:
                N_1 += 1
            elif target[1] == 1:
                N_2 += 1
            else:
                raise ValueError("Something is wrong")

            t = self.y(X_test[i], self.f, self.kernel)
            t_argmax = np.argmax(t)
            target = np.argmax(target)
            if t_argmax == target:
                if t_argmax == 0:
                    correct_1 += 1
                elif t_argmax == 1:
                    correct_2 += 1
                else:
                    raise ValueError("Something is wrong")

        N_test = N_1 + N_2
        assert N_test == len(T_test)
        acc_1 = correct_1 / N_1
        acc_2 = correct_2 / N_2
        accuracy = (correct_1 + correct_2) / N_test
        return accuracy, acc_1, acc_2


class LS:
    """
    Least squares.
    X: List containing lists x
    T: List containing target values
    """

    def __init__(self, X, T):
        self.T = np.array(T, dtype=int)
        self.X = np.row_stack(X)
        self.X = np.insert(self.X, 0, 1, axis=1)  # Add first column of ones
        self.X = self.X.astype(int)
        self.X_T = self.X.transpose()
        self.gram = np.dot(self.X_T, self.X)  # Gram matrix
        self.weights = None
        self.lam = None  # lambda

    def train(self, lam):
        """
        Calculates weights using the specified lambda.
        """
        self.lam = lam
        gram_inv = np.linalg.inv(self.gram + lam * np.identity(len(self.X_T)))
        MP_inverse = np.dot(gram_inv, self.X_T)
        self.weights = np.dot(MP_inverse, self.T)

    def y(self, x, w):
        """
        Calculate output value of input x, using the weights w.
        Returns +1 (boy) or -1 (girl)
        """
        t = w[0] + np.dot(x, w[1:])
        return t

    def test(self, X_test, T_test, show_imgs=False, shape=(48, 32, 3)):
        """
        Returns total accuracy, accuracy -1 class, accuracy +1 class
        """
        correct_1 = 0  # nr of correct predictions of the 1 class
        N_1 = 0  # nr of test inputs of the 1 class
        correct_2 = 0  # nr of correct predictions of the 2 class
        N_2 = 0  # nr of test inputs of the 2 class

        for i in range(len(X_test)):
            target = T_test[i]
            if target[0] == 1:
                N_1 += 1
            elif target[1] == 1:
                N_2 += 1
            else:
                raise ValueError("Something is wrong")

            t = self.y(X_test[i], self.weights)
            t_argmax = np.argmax(t)
            target = np.argmax(target)
            correct = 0
            v = ["boy.", "girl."]
            if t_argmax == target:
                correct = 1
                if t_argmax == 0:
                    correct_1 += 1
                elif t_argmax == 1:
                    correct_2 += 1
                else:
                    raise ValueError("Something is wrong")

            if show_imgs is True:
                fig = plt.figure()
                if correct == 1:
                    plt.title("Correctly classified as " + v[int(t_argmax)] +
                              "\nBoy: " + str(np.round(t[0], 2)) + ", Girl: " + str(np.round(t[1], 2)))
                else:
                    plt.title("Wrongly classified as " + v[int(t_argmax)] +
                              "\nBoy: " + str(np.round(t[0], 2)) + ", Girl: " + str(np.round(t[1], 2)))
                plt.imshow(np.reshape(X_test[i], shape))
                button = plt.waitforbuttonpress(-1)
                if button:
                    plt.close(fig)
                else:
                    plt.close(fig)
                    show_imgs = False

        N_test = N_1 + N_2
        assert N_test == len(T_test)
        acc_1 = correct_1 / N_1
        acc_2 = correct_2 / N_2
        accuracy = (correct_1 + correct_2) / N_test
        return accuracy, acc_1, acc_2


def averageDistance(X):
    d = 0
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            d += np.linalg.norm(X[i] - X[j])
    return d / len(X)

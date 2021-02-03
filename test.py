import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pickle
import sklearn.svm
import sklearn.metrics
from models import *

pickles_path = ""

# Scaled inputs 255
X_train = pickle.load(open(pickles_path + "/Train/tr_input0.p", "rb"))
T_train = pickle.load(open(pickles_path + "/Train/tr_labels0.p", "rb"))
X_test = pickle.load(open(pickles_path + "/Test/test_input0.p", "rb"))
T_test = pickle.load(open(pickles_path + "/Test/test_labels0.p", "rb"))

nrTests = 4

for i in range(nrTests):

    # LS
    E = LS(X_train, T_train)
    lam = 6e5
    E.train(lam)
    accuracy, acc_boy, acc_girl = E.test(X_test, T_test)
    print("Least squares:")
    print("Girl's: " + str(acc_girl) + "\nBoy's: " + str(acc_boy))
    print("Total accuracy: " + str(accuracy) + "\n")

    # Kernel LS
    kernel = "rbf"
    sigma = averageDistance(X_train)
    lam = 1e-2

    E = KRR(X_train, T_train, kernel, s=sigma)
    E.train(lam)
    accuracy, acc_boy, acc_girl = E.test(X_test, T_test)
    print("Kernel ridge regression:")
    print("Girl's: " + str(acc_girl) + "\nBoy's: " + str(acc_boy))
    print("Total accuracy: " + str(accuracy) + "\n")

    # ELM
    nrNodes = 1000
    lam = 1e5
    E = ELM(X_train, T_train)
    act = "softplus"    # Activation function relu
    E.train(act, lam, nrNodes)
    accuracy, acc_boy, acc_girl = E.test(X_test, T_test)
    print("Extreme learning machine:")
    print("Girl's: " + str(acc_girl) + "\nBoy's: " + str(acc_boy))
    print("Total accuracy: " + str(accuracy) + "\n")

    # SVM
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    sigma = averageDistance(X_train)
    gamma = 1 / sigma
    T_new = []
    for t in T_train:
        T_new.append(np.argmax(t))
    T_train = np.asarray(T_new)
    T_new = []
    for t in T_test:
        T_new.append(np.argmax(t))
    T_test = np.asarray(T_new)
    SVM = sklearn.svm.SVC(kernel='rbf', C=100, gamma=gamma)
    SVM.fit(X_train, T_train)
    T_pred = SVM.predict(X_test)
    accuracy = sklearn.metrics.accuracy_score(T_test, T_pred)
    print("SVM:")
    print(accuracy)

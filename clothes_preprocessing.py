import cv2
import numpy as np
import pickle
import sklearn.utils

N = 1000
N_class = int(N / 2)
N_train = int(0.75 * N)
N_train_class = int(N_train / 2)
N_test = int(0.25 * N)
N_test_class = int(N_test / 2)

# Size of images: 768x1152
resize_images = 24
scale_input = 255

# Enter the file path for boy-images and girl-images
file_path_boy = ""
file_path_girl = ""

# Determine new width and height for the images
image = cv2.imread(file_path_boy + "/1.jpeg")
height = len(image)
width = len(image[0])
height = int(height / resize_images)
width = int(width / resize_images)

nrShuffles = 10

for n in range(nrShuffles):
    print(n)
    X_boy = []
    X_girl = []

    # Boy's
    for i in range(1, N_class + 1):
        image = cv2.imread(file_path_boy + str(i) + ".jpeg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (width, height),
                           interpolation=cv2.INTER_AREA)
        image = image / scale_input
        X_boy.append(image.ravel())

    X_boy = sklearn.utils.shuffle(X_boy)    # Shuffle the images

    # Girl's
    for i in range(1, N_class + 1):
        image = cv2.imread(file_path_girl + str(i) + ".jpeg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (width, height),
                           interpolation=cv2.INTER_AREA)
        image = image / scale_input
        X_girl.append(image.ravel())

    X_girl = sklearn.utils.shuffle(X_girl)    # Shuffle the images

    X_train = X_boy[:N_train_class]
    X_train.extend(X_girl[:N_train_class])

    X_test = X_boy[N_train_class:]
    X_test.extend(X_girl[N_train_class:])

    T_train = [[1, 0] for i in range(N_train_class)]    # [1, 0] : boy
    T_train.extend([[0, 1] for i in range(N_train_class)])    # [0, 1] : girl

    T_test = [[1, 0] for i in range(N_test_class)]
    T_test.extend([[0, 1] for i in range(N_test_class)])

    X_train, T_train = sklearn.utils.shuffle(X_train, T_train)
    X_test, T_test = sklearn.utils.shuffle(X_test, T_test)

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    T_train = np.asarray(T_train)
    T_test = np.asarray(T_test)

    # pickle.dump(X_train, open("../Pickles/Train/tr_input%d.p" % n, "wb"))  # wb = write binary
    # pickle.dump(T_train, open("../Pickles/Train/tr_labels%d.p" % n, "wb"))

    # pickle.dump(X_test, open("../Pickles/Test/test_input%d.p" % n, "wb"))
    # pickle.dump(T_test, open("../Pickles/Test/test_labels%d.p" % n, "wb"))

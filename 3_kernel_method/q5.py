# EECS 545 HW3 Q5
# Your name: Aabhaas Vaish (aabhaas@umich.edu)

# Install scikit-learn package if necessary:
# pip install -U scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC


def readMatrix(filename: str):
    # Use the code below to read files
    with open(filename, 'r') as fd:
        hdr = fd.readline()
        rows, cols = [int(s) for s in fd.readline().strip().split()]
        tokens = fd.readline().strip().split()
        matrix = np.zeros((rows, cols))
        Y = []
        for i, line in enumerate(fd):
            nums = [int(x) for x in line.strip().split()]
            Y.append(nums[0])
            kv = np.array(nums[1:])
            k = np.cumsum(kv[:-1:2])
            v = kv[1::2]
            matrix[i, k] = v
        return matrix, tokens, np.array(Y)


def evaluate(output, label) -> float:
    # Use the code below to obtain the accuracy of your algorithm
    error = float((output != label).sum()) * 1. / len(output)
    print('Error: {:2.4f}%'.format(100 * error))

    return error


def get_num_support_vectors(svm, data) -> int:
    values = svm.decision_function(data)
    return np.count_nonzero(np.abs(values) <= 1 + 1e-15)


def main():
    # Load files
    # Note 1: tokenlists (list of all tokens) from MATRIX.TRAIN and MATRIX.TEST are identical
    # Note 2: Spam emails are denoted as class 1, and non-spam ones as class 0.
    dataMatrix_train, tokenlist, category_train = readMatrix('q5_data/MATRIX.TRAIN')
    dataMatrix_test, tokenlist, category_test = readMatrix('q5_data/MATRIX.TEST')

    # Train
    svm_classifier = LinearSVC(random_state=0, max_iter=100000)
    svm_classifier.fit(dataMatrix_train, category_train)
    
    # Test and evluate
    prediction = svm_classifier.predict(dataMatrix_test)
    print("--------------------------------------")
    print("Using the training set MATRIX.TRAIN (Part A)")
    evaluate(prediction, category_test)
    print("Number of Support Vectors used:", get_num_support_vectors(svm_classifier, dataMatrix_train))
    print("--------------------------------------\n")
    
    print("(Part B)")
    # Learning Curve
    sizes = [50, 100, 200, 400, 800, 1400]
    test_errors = []
    s_svm = LinearSVC(random_state=0, max_iter=100000)
    for size in sizes:
        dataMatrix_train_size, tlist, category_train_size = readMatrix('q5_data/MATRIX.TRAIN.' + str(size))
        s_svm.fit(dataMatrix_train_size, category_train_size)
        prediction = s_svm.predict(dataMatrix_test)
        print("Using the training set of size", size)
        test_errors.append(100*evaluate(prediction, category_test))
        print("Number of Support Vectors:", get_num_support_vectors(s_svm, dataMatrix_train_size))
        print("")

    plt.clf()
    plt.xlabel('Size of Training Set')
    plt.ylabel('Test Error (%)')
    plt.plot(sizes, test_errors, 'o-', label='Test Error')
    plt.legend()
    plt.grid()
    plt.title('Test Error (Linear SVM) v/s Training Set Size')
    plt.savefig('q5_b.png', dpi=250)

    
if __name__ == '__main__':
    main()
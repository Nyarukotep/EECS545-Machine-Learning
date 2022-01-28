import numpy as np
import matplotlib.pyplot as plt

def readMatrix(file):
    # Use the code below to read files
    fd = open(file, 'r')
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

def nb_train(matrix, category):
    # Implement your algorithm and return 
    state = {}
    N = matrix.shape[1]
    
    ############################
    # Implement your code here #
    ############################
    
    return state

def nb_test(matrix, state):
    # Classify each email in the test set (each row of the document matrix) as 1 for SPAM and 0 for NON-SPAM
    output = np.zeros(matrix.shape[0])
    
    ############################
    # Implement your code here #
    ############################
    
    return output

def evaluate(output, label):
    # Use the code below to obtain the accuracy of your algorithm
    error = (output != label).sum() * 1. / len(output)
    print('Error: {:2.4f}%'.format(100*error))

def main():
    # Note1: tokenlists (list of all tokens) from MATRIX.TRAIN and MATRIX.TEST are identical
    # Note2: Spam emails are denoted as class 1, and non-spam ones as class 0.
    # Note3: The shape of the data matrix (document matrix): (number of emails) by (number of tokens)

    # Load files
    dataMatrix_train, tokenlist, category_train = readMatrix('q4_data/MATRIX.TRAIN')
    dataMatrix_test, tokenlist, category_test = readMatrix('q4_data/MATRIX.TEST')

    # Train
    state = nb_train(dataMatrix_train, category_train)

    # Test and evluate
    prediction = nb_test(dataMatrix_test, state)
    evaluate(prediction, category_test)

if __name__ == "__main__":
    main()
        

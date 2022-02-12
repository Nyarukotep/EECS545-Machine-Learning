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
    spam = matrix[category == 1, :]
    nospam = matrix[category == 0, :]
    state['spamlap'] = (spam.sum(axis = 0) + 1) / (np.sum(spam.sum(axis = 1)) + N)
    state['nospamlap'] = (nospam.sum(axis = 0) + 1) / (np.sum(nospam.sum(axis = 1)) + N)
    state['p'] = spam.shape[0]/(spam.shape[0]+nospam.shape[0])
    ############################
    return state

def nb_test(matrix, state):
    # Classify each email in the test set (each row of the document matrix) as 1 for SPAM and 0 for NON-SPAM
    output = np.zeros(matrix.shape[0])
    
    ############################
    phi = state['p']
    output[matrix @ np.log(state['spamlap']) + np.log(phi) > matrix @ np.log(state['nospamlap']) + np.log(1-phi)] = 1
    ############################
    return output

def evaluate(output, label):
    # Use the code below to obtain the accuracy of your algorithm
    error = (output != label).sum() * 1. / len(output)
    print('Error: ', 100*error, '%')
    return error

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
    list = np.argsort(state['spamlap']/state['nospamlap'])[-5:]
    print('The 5 tokens that are most indicative of the SPAM class: ')
    for i in range(5):
        print(tokenlist[list[4-i]])
    train_sizes = [50, 100, 200, 400, 800, 1400]
    errors = np.ones(6)
    for i in range(6):
        file = 'q4_data/MATRIX.TRAIN.'+str(train_sizes[i])
        dataMatrix_train, tokenlist, category_train = readMatrix(file)
        state = nb_train(dataMatrix_train, category_train)
        prediction = nb_test(dataMatrix_test, state)
        errors[i] = evaluate(prediction, category_test)
    plt.plot(train_sizes, errors*100)
    plt.xlim((0, 1500))
    plt.ylim((0, 4))
    plt.xlabel('Training size')
    plt.ylabel('Error %')
    plt.show()
if __name__ == "__main__":
    main()
        

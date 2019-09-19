import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    file_x_test_data = pd.read_csv('x_test.csv', header=None)
    file_y_test_data = pd.read_csv('y_test.csv', header=None)
    file_x_train_data = pd.read_csv('x_train.csv', header=None)
    file_y_train_data = pd.read_csv('y_train.csv', header=None)

    x_test_value = [x[0] for x in file_x_test_data.values]
    y_test_value = [y[0] for y in file_y_test_data.values]
    x_train_value = [x[0] for x in file_x_train_data.values]
    y_train_value = [y[0] for y in file_y_train_data.values]

    test_set = pd.DataFrame({'x': x_test_value, 'y': y_test_value}, index=None)
    train_set = pd.DataFrame({'x': x_train_value, 'y': y_train_value}, index=None)

    # Q1(c)
    plt.figure(1)
    plt.scatter(train_set['x'], train_set['y'])
    plt.title('Scatter Plot of Training Set')
    plt.xlabel('x_train')
    plt.ylabel('y_train')
    plt.show()

    # Q1(d)
    weight = {}
    degree = [1, 2, 3, 7, 10]
    color = ['r', 'b', 'c', 'g', 'm']
    for deg in degree:
        weight[deg] = get_weight(train_set, deg)

    plt.figure(2, (15, 10), 200)
    plt.scatter(train_set['x'], train_set['y'], label='training set')
    for i in range(5):
        y_data = np.dot(get_data_matrix([x / 10 for x in range(101)], degree[i]), weight[degree[i]])
        plt.plot([x / 10 for x in range(101)], y_data, color=color[i], label='degree: ' + str(degree[i]))
    plt.legend()
    plt.title('Fitting Curves')
    plt.xlabel('x_train')
    plt.ylabel('y_train')

    print('Q1(d)')
    for key in weight.keys():
        print('degree =', key, 'weight =', weight[key])
    print(' ')

    # Q1(e)
    mse_train = []
    for d in degree:
        mse_train.append(get_mse(train_set, d, weight[d]))
    plt.figure(3)
    plt.plot(degree, mse_train, marker='o')
    plt.title('Training Error vs. polynomial degree')
    plt.xlabel('degree')
    plt.ylabel('MSE')
    print('Q1(e)')
    print('The best model based on training set is when degree =', degree[mse_train.index(min(mse_train))])
    print('MSE =', min(mse_train))
    print('')

    # Q1(f)
    mse_test = []
    for d in degree:
        mse_test.append(get_mse(test_set, d, weight[d]))
    plt.figure(4)
    plt.plot(degree, mse_test, marker='o')
    plt.title('Test Error vs. polynomial degree')
    plt.xlabel('degree')
    plt.ylabel('MSE')
    print('Q1(f)')
    print('The best model based on test set is when degree =', degree[mse_test.index(min(mse_test))])
    print('MSE =', min(mse_test))
    print('')

    # Q1(g)
    fix_degree = 7
    l2_lambda = [1e-5, 1e-3, 0.1, 1, 10]
    l2_weight = {}
    l2_x_train = get_data_matrix(train_set['x'], fix_degree)
    xTx = np.transpose(l2_x_train).dot(l2_x_train)
    I = np.identity(xTx.shape[0])

    print('Q1(g)')
    for lam in l2_lambda:
        l2_weight[lam] = np.linalg.inv(lam * I + xTx).dot(np.transpose(l2_x_train)).dot(train_set['y'])
        print('lambda =', lam, 'weight =', l2_weight[lam])

    # Q1(h)
    l2_mse_train = []
    l2_mse_test = []
    for lam in l2_lambda:
        l2_mse_test.append(get_mse(test_set, fix_degree, l2_weight[lam]))
        l2_mse_train.append(get_mse(train_set, fix_degree, l2_weight[lam]))

    plt.figure(5)
    plt.plot([math.log10(x) for x in l2_lambda], l2_mse_test, marker='x', color='b', label='test')
    plt.plot([math.log10(x) for x in l2_lambda], l2_mse_train, marker='o', color='r', label='train')
    plt.yticks(np.arange(0.05, 0.17, 0.01))
    plt.xlabel('log(lambda)')
    plt.ylabel('MSE')
    plt.title('Train and Test MSE vs. log(lambda)')
    plt.legend()

    plt.show()


# using np.linalg.inv to get pinv
def get_pseudo_inverse(matrix):
    return np.linalg.inv(np.transpose(matrix).dot(matrix)).dot(np.transpose(matrix))


# return the data matrix of φ(x)
def get_data_matrix(data_set, polynomial_degree) -> list:
    x_matrix = []
    for x in data_set:
        row = [1]
        for d in range(1, polynomial_degree + 1):
            row.append(x ** d)
        x_matrix.append(row)
    return x_matrix


# by using pseudo inverse to get the weight of regression
def get_weight(data_set, polynomial_degree) -> list:
    data_matrix = get_data_matrix(data_set['x'], polynomial_degree)
    return np.dot(get_pseudo_inverse(data_matrix), data_set['y'])


# calculate MSE of a model
def get_mse(data, degree, weight):
    data_matrix = get_data_matrix(data['x'], degree)
    MSE = 0
    y = np.dot(data_matrix, weight)
    for i in range(len(data['y'])):
        MSE += (data['y'][i] - y[i]) ** 2
    return MSE / len(data['y'])


if __name__ == '__main__':
    main()

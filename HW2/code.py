import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.linear_model.logistic import LogisticRegression


def main():
    # read all csv file
    file_x_test_data = pd.read_csv('Xtest.csv', header=None)
    file_y_test_data = pd.read_csv('ytest.csv', header=None)
    file_x_train_data = pd.read_csv('Xtrain.csv', header=None)
    file_y_train_data = pd.read_csv('ytrain.csv', header=None)

    file_x_test_value = file_x_test_data.values
    file_y_test_value = file_y_test_data.values
    file_x_train_value = file_x_train_data.values
    file_y_train_value = file_y_train_data.values

    columns = [str(x) for x in range(1, 56)]

    # build test set and train/validation aet
    test_frame = pd.DataFrame(file_x_test_value, columns=columns + ['56', '57'])
    test_frame = test_frame.drop(['56', '57'], axis=1)
    test_frame['y'] = file_y_test_value
    train_validation_frame = pd.DataFrame(file_x_train_value, columns=columns + ['56', '57'])
    train_validation_frame = train_validation_frame.drop(['56', '57'], axis=1)
    train_validation_frame['y'] = file_y_train_value

    # set lambda value
    lmbd = [10 ** x for x in np.arange(-2, 2, 0.05)]

    # Question A:
    print('standardized result:')
    run_LR(lmbd, 'std', train_validation_frame, test_frame, columns)

    print('log transform result:')
    run_LR(lmbd, 'log', train_validation_frame, test_frame, columns)

    print('binarized result:')
    run_LR(lmbd, 'bi', train_validation_frame, test_frame, columns)

    # Additional Question:
    run_addition_question(test_frame)


def run_addition_question(test_frame):
    # build label0 test set and label1 test set
    set_0, set_1 = preprocess_data(test_frame)

    # question 1
    plot_scatter(set_0, set_1)

    # question 2
    plot_3d_hist(set_0)

    # question 3
    plot_3d_hist(set_1)

    # show all figures
    plt.show()


# using for additional question to binarize test set
def preprocess_data(test_frame):
    function = preprocessing.Binarizer(threshold=0)
    row_test_data = function.transform(test_frame)
    label_0 = []
    label_1 = []
    for ele in row_test_data:
        if ele[-1]:
            label_1.append([sum(ele[:48]), sum(ele[48:54]), 1])
        else:
            label_0.append([sum(ele[:48]), sum(ele[48:54]), 0])
    set_0 = pd.DataFrame(label_0, columns=['x', 'y', 'label'])
    set_1 = pd.DataFrame(label_1, columns=['x', 'y', 'label'])
    return[set_0, set_1]


# draw a scatter plot of all testing points
def plot_scatter(set_0, set_1):
    plt.figure(np.random.randint(1e10))
    plt.scatter(set_0['x'], set_0['y'], label='0: non-spam', color='r', marker='o')
    plt.scatter(set_1['x'], set_1['y'], label='1: spam', color='b', marker='x')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('A scatter plot of all testing points')
    plt.legend()


# generate a 3D histogram
def plot_3d_hist(set_i):
    ax = Axes3D(plt.figure(np.random.randint(1e10)))
    hist, xedge, yedge = np.histogram2d(set_i['x'], set_i['y'], bins=30, range=[[0, 30],[0, 6]])
    xpos, ypos = np.meshgrid(xedge[:-1]+0.25, yedge[:-1]+0.25, indexing='ij')
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0
    dx = dy = 0.5 * np.ones_like(zpos)
    dz = hist.ravel()
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', color='r')
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_zlabel('frequency')
    if set_i['label'][0]:
        plt.title('spam')
    else:
        plt.title('non-spam')


# Logistic regression / and print result
def run_LR(lmbd, preprocess_method, train_validation_frame, test_frame, columns):
    result = {}
    minimum_cv_error = 1
    for lamda in lmbd:
        cv_accuracy = []
        for times in range(5):
            # generate train set and validation set
            train_frame, validation_frame = split_train_set(train_validation_frame, columns)
            train_set, validation_set, test_set = transform(preprocess_method, train_frame, validation_frame,
                                                            test_frame, columns)

            # Logistic regression
            L2_classify = LogisticRegression(C=1 / lamda, penalty='l2', solver='newton-cg')
            L2_classify.fit(train_set.drop(['y'], axis=1), train_set['y'])

            # cv for model selection
            cv_result_list = cross_val_score(L2_classify, validation_set.drop(['y'], axis=1), validation_set['y'], cv=5)
            cv_accuracy.append(np.mean(cv_result_list))
        result[lamda] = 1 - np.mean(cv_accuracy)
        minimum_cv_error = min(minimum_cv_error, result[lamda])

    # print all lambda and cv error
    for key in result.keys():
        print('when lambda =', key, 'cv error rate =', result[key])

    # select the best model
    for key in result.keys():
        if result[key] == minimum_cv_error:
            print('the best model is when lambda =', key)
            print('cv error rate =', minimum_cv_error)

            train_set, no_use_set, test_set = transform(preprocess_method, train_validation_frame,
                                                        train_validation_frame,
                                                        test_frame, columns)
            L2_classify = LogisticRegression(C=1 / key, penalty='l2', solver='newton-cg')
            L2_classify.fit(train_set.drop(['y'], axis=1), train_set['y'])

            # print test error
            print('test error =', 1 - L2_classify.score(test_set.drop(['y'], axis=1), test_set['y']))

            # print train error for whole train set
            print('train error =', 1 - L2_classify.score(train_set.drop(['y'], axis=1), train_set['y']))
            break
    print('')


# log transform function
def log01p(x):
    return np.log(x + 0.1)


# preprocess the data by different method
def transform(type, train_frame, validation_frame, test_frame, columns):
    test_frame_X = test_frame.drop(['y'], axis=1)
    test_frame_Y = test_frame['y']
    train_frame_X = train_frame.drop(['y'], axis=1)
    train_frame_Y = train_frame['y']
    validation_frame_X = validation_frame.drop(['y'], axis=1)
    validation_frame_Y = validation_frame['y']

    if type == 'log':
        function = preprocessing.FunctionTransformer(log01p, validate=False)
    elif type == 'bi':
        function = preprocessing.Binarizer(threshold=0)
    elif type == 'std':
        function = preprocessing.StandardScaler().fit(train_frame_X)

    test_data_X = function.transform(test_frame_X)
    train_data_X = function.transform(train_frame_X)
    validation_data_X = function.transform(validation_frame_X)

    test_set = pd.DataFrame(test_data_X, columns=columns)
    test_set['y'] = test_frame_Y.values
    train_set = pd.DataFrame(train_data_X, columns=columns)
    train_set['y'] = train_frame_Y.values
    validation_set = pd.DataFrame(validation_data_X, columns=columns)
    validation_set['y'] = validation_frame_Y.values
    return [train_set, validation_set, test_set]


# split the whole train file to train set and validation set
def split_train_set(train_val_frame, columns):
    N = train_val_frame.values.shape[0]
    data = train_val_frame.values[:]
    np.random.shuffle(data)
    train_number = int(np.round(N * 0.7))
    validation_number = N - train_number
    train_set_data = data[:train_number]
    validation_set_data = data[-validation_number:]
    train_frame = pd.DataFrame(train_set_data, columns=columns + ['y'])
    validation_frame = pd.DataFrame(validation_set_data, columns=columns + ['y'])
    return [train_frame, validation_frame]


if __name__ == '__main__':
    main()

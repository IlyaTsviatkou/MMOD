import math

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as sts

def generate_new_variables(N, n):
    array = []
    while True:
        temp = np.random.randint(1, n)
        if temp not in array:
            array.append(temp)
            if len(array) == N:
                return np.sort(np.array(array))

def get_delta(XY):
    delta = []
    for i in range(np.shape(XY)[0]):
        delta = np.append(delta, sum(XY[i]))
    return delta

def F(delta):
    F = np.cumsum(delta)
    return np.array(F)

def F_x(XY, delta):
    F_x = []
    N, M = np.shape(XY)
    for n in range(N):
        F_x.append([])
        temp_delta = np.cumsum(XY[n])
        for m in range(M):
            F_x[-1].append(temp_delta[m] / delta[n])
    return np.array(F_x)

def generateDiscreteSV(n, X, Y, F, F_x):
    discrete_XY = []

    for i in range(n):
        discrete_XY.append([])
        x, y = np.random.uniform(size=2)

        x_index = np.searchsorted(F, x)
        y_index = np.searchsorted(F_x[x_index], y)

        discrete_XY[-1].append(X[x_index])
        discrete_XY[-1].append(Y[y_index])
    return discrete_XY

def show_chart(t_matrix, e_matrix, discrete_2d, X, Y):
    ax = plt.axes()
    sns.heatmap(t_matrix, xticklabels=Y, yticklabels=X, annot=True, vmin=0, vmax=1, cmap='coolwarm', ax=ax)
    ax.set_title('Theoretical_matrix')
    plt.show()

    ax = plt.axes()
    sns.heatmap(e_matrix, xticklabels=Y, yticklabels=X, annot=True, vmin=0, vmax=1, cmap='coolwarm', ax=ax)
    ax.set_title('Empirical_matrix')
    plt.show()

    count = len(discrete_2d)

    x_elements = [item[0] for item in discrete_2d]
    values = [x_elements.count(x) / count for x in X]
    fig, ax = plt.subplots(1, 1)
    ax.bar(X, values, width=0.1)
    ax.set_title('Empirical_matrix_X')
    plt.show()

    y_elements = [item[1] for item in discrete_2d]
    values = [y_elements.count(y) / count for y in Y]
    fig1, ax1 = plt.subplots(1, 1)
    ax1.bar(Y, values, width=0.1)
    ax1.set_title('Empirical_matrix_Y')
    plt.show()

def empirical_probability(discrete_2d, X, Y):
    empirical_matrix = np.zeros((X.size, Y.size))

    for x, y in discrete_2d:
        x_index = np.where(X == x)
        y_index = np.where(Y == y)
        empirical_matrix[x_index, y_index] = discrete_2d.count([x, y]) / len(discrete_2d)

    return np.array(empirical_matrix)

##### Calculations
def calculate_t_expectation(matrix, X, Y):
    delta_x = np.sum(matrix, axis=1)
    delta_y = np.sum(matrix, axis=0)
    expectation_x = sum(delta_x * X)
    expectation_y = sum(delta_y * Y)
    return expectation_x, expectation_y

def calculate_t_variance(matrix, X, Y, m_x, m_y):
    delta_x = np.sum(matrix, axis=1)
    delta_y = np.sum(matrix, axis=0)
    sq_expectation_x = sum(delta_x * (X ** 2))
    sq_expectation_y = sum(delta_y * (Y ** 2))
    variance_x = sq_expectation_x - m_x ** 2
    variance_y = sq_expectation_y - m_y ** 2
    return variance_x, variance_y

def calculate_e_expectation(discrete_2d):
    delta_x = sum([item[0] for item in discrete_2d])
    delta_y = sum([item[1] for item in discrete_2d])
    count = len(discrete_2d)
    expectation_x = delta_x / count
    expectation_y = delta_y / count
    return expectation_x, expectation_y

def calculate_e_variance(discrete_2d, m_x, m_y):
    delta_x = sum([(item[0] - m_x) ** 2 for item in discrete_2d])
    delta_y = sum([(item[1] - m_y) ** 2 for item in discrete_2d])
    count = len(discrete_2d)
    variance_x = delta_x / (count - 1)
    variance_y = delta_y / (count - 1)
    return variance_x, variance_y

def calculate_e_interval_estimations_expectation(discrete_2d, m_x, m_y, D_x, D_y):
    count = len(discrete_2d)
    tt = sts.t(count - 1)
    arr = tt.rvs(1000000)

    delta = sts.mstats.mquantiles(arr, prob=0.95) * math.sqrt(D_x / (count - 1))
    interval_estimations_x = m_x - delta, m_x + delta

    delta = sts.mstats.mquantiles(arr, prob=0.95) * math.sqrt(D_y / (count - 1))
    interval_estimations_y = m_y - delta, m_y + delta

    return interval_estimations_x, interval_estimations_y


def calculate_e_interval_estimations_variance(discrete_2d, D_x, D_y):
    count = len(discrete_2d)
    tt = sts.chi2(count - 1)
    arr = tt.rvs(1000000)

    delta = sts.mstats.mquantiles(arr, prob=[0.01, 0.99])

    interval_x = (count * D_x / delta[1], count * D_x / delta[0])
    interval_y = (count * D_y / delta[1], count * D_y / delta[0])

    return interval_x, interval_y


def calculate_correlation_coefficient(discrete_2d, m_x, m_y):
    numerator = sum([((item[0] - m_x) * (item[1] - m_y)) for item in discrete_2d])
    temp = [((item[0] - m_x) ** 2, (item[1] - m_y) ** 2) for item in discrete_2d]
    denominator = math.sqrt(sum([item[0] for item in temp]) * sum([item[1] for item in temp]))
    correlation_coefficient = numerator / denominator
    return correlation_coefficient


def correlationDiscrete(X, Y, probability_matrix, m_x, m_y, D_x, D_y):
    cov = 0
    for i in range(len(X)):
        for j in range(len(Y)):
            cov = cov + (X[i] * Y[j] * probability_matrix[i][j])

    cov -= m_x * m_y
    correlation = cov / np.sqrt(D_x * D_y)
    return correlation


##### MAIN
if __name__ == "__main__":

    N, M = 8, 8
    X = generate_new_variables(N, 10)
    print('X = ', X)
    Y = generate_new_variables(M, 10)
    print('Y = ', Y)

    matrix = np.random.dirichlet(np.ones(N * M))
    temp = []
    for n in range(N):
        temp.append(matrix[M * n:M * (n + 1)])
    XY = np.array(temp)

    print('theoretical_matrix : \n{}\n'.format(XY))

    delta_x = get_delta(XY)
    print("delta_x : \n{0}\n".format(delta_x))

    print("Total probability : \n{0}\n".format(sum(delta_x)))

    print('Construct f. Distribution of DSV for variable X\n')
    F = F(delta_x)
    print('F : \n{}\n'.format(F))
    F_x = F_x(XY, delta_x)
    print('F_x : \n{}\n'.format(F_x))

    discrete_SV = generateDiscreteSV(10000, X, Y, F, F_x)
    #//////////////////////////////////////////
    print('Find the empirical distribution matrix\n')
    empirical_probability = empirical_probability(discrete_SV, X, Y)
    print('empirical_matrix : \n{}\n'.format(empirical_probability))

    show_chart(XY, empirical_probability, discrete_SV, X, Y)
    #//////////////////////////////////////////
    print('Find point estimates\n')
    t_expectation = calculate_t_expectation(XY, X, Y)
    print('theoretical_expectation : \n{}\n'.format(t_expectation))

    e_expectation = calculate_e_expectation(discrete_SV)
    print('empirical_expectation : \n{}\n'.format(e_expectation))

    t_variance = calculate_t_variance(XY, X, Y, t_expectation[0], t_expectation[1])
    print('theoretical_variance : \n{}\n'.format(t_variance))

    e_variance = calculate_e_variance(discrete_SV, e_expectation[0], e_expectation[1])
    print('empirical_variance : \n{}\n'.format(e_variance))
    #//////////////////////////////////////////
    print('Find interval estimates\n')
    interval_estimations_expectation = calculate_e_interval_estimations_expectation(discrete_SV, e_expectation[0], e_expectation[1], e_variance[0], e_variance[1])
    print('interval_estimations_x : \n{0}\ninterval_estimations_y : \n{1}\n'.format(interval_estimations_expectation[0], interval_estimations_expectation[1]))

    interval_estimations_variance = calculate_e_interval_estimations_variance(discrete_SV, e_variance[0], e_variance[1])
    print('interval_estimations_x : \n{0}\ninterval_estimations_y : \n{1}\n'.format(interval_estimations_variance[0], interval_estimations_variance[1]))
    #//////////////////////////////////////////
    print('Find the correlation coefficient\n')

    t_correlation_coefficient = correlationDiscrete(X, Y, XY, t_expectation[0], t_expectation[1], t_variance[0], t_variance[1])
    print('theoretical_correlation_coefficient : \n{0}\n'.format(t_correlation_coefficient))

    e_correlation_coefficient = correlationDiscrete(X, Y, empirical_probability, e_expectation[0], e_expectation[1], e_variance[0], e_variance[1])
    print('empirical_correlation_coefficient : \n{0}\n'.format(e_correlation_coefficient))
    #/////////////////////////////////////////
    if interval_estimations_expectation[0][0] < e_expectation[0] < interval_estimations_expectation[0][1]:
        print('point probability expectation mate X fell into the interval gap with probability 0.99')
    else:
        print('point probability of the expectation mate X did not fall into the interval gap with a probability of 0.99')
    if interval_estimations_variance[0][0] < e_variance[0] < interval_estimations_variance[0][1]:
        print('point probability of the variance X falls into the interval gap with a probability of 0.99')
    else:
        print('point probability of the variance X did not fall into the interval gap with a probability of 0.99')
    if interval_estimations_expectation[1][0] < e_expectation[1] < interval_estimations_expectation[1][1]:
        print('point probability of expectation mate Y fell into the interval gap with probability 0.99')
    else:
        print('point probability of the expectation mate Y did not fall into the interval gap with a probability of 0.99')
    if interval_estimations_variance[1][0] < e_variance[1] < interval_estimations_variance[1][1]:
        print('point probability of the variance Y falls into the interva gap with a probability of 0.99')
    else:
        print('point probability of the variance Y did not fall into the interval gap with a probability of 0.99')

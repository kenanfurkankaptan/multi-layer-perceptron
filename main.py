import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt

from data_generator import *
from multi_layer_neural_network import *


original_x = np.zeros(4).tolist()
original_y = np.zeros(4).tolist()

# example data are in 5x10, dont change the dimentions
# Letter 'O' in array
original_y[0] = np.array([0.9, 0.1, 0.1, 0.1])
original_x[0] = np.array([
    np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1]),
    np.array([0.1, 0.1, 0.1, 0.9, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1]),
    np.array([0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1]),
    np.array([0.1, 0.1, 0.1, 0.9, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1]),
    np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1])
])

# Letter 'L' in array
original_y[1] = np.array([0.1, 0.9, 0.1, 0.1])
original_x[1] = np.array([
    np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    np.array([0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
])

# Letter 'N' in array
original_y[2] = np.array([0.1, 0.1, 0.9, 0.1])
original_x[2] = np.array([
    np.array([0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1]),
    np.array([0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1]),
    np.array([0.1, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.1, 0.1]),
    np.array([0.1, 0.1, 0.9, 0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.1]),
    np.array([0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1])
])

# Letter 'E' in array
original_y[3] = np.array([0.1, 0.1, 0.1, 0.9])
original_x[3] = np.array([
    np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1]),
    np.array([0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1]),
    np.array([0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1])
])


data_generator(original_x, original_y, 20, 12)

# define training and test lists
train_x = np.zeros(20).tolist()
train_y = np.zeros(20).tolist()

test_x = np.zeros(12).tolist()
test_y = np.zeros(12).tolist()

# load data arrays
train_x = loadtxt('train_x.csv', delimiter=',')
train_y = loadtxt('train_y.csv', delimiter=',')
test_x = loadtxt('test_x.csv', delimiter=',')
test_y = loadtxt('test_y.csv', delimiter=',')


train_error = 0
iteration = 0
test_error = 0
predicted_true = 0
predicted_false = 0

# train and test 20 times and take average
number_of_repeat = 20
for i in range(number_of_repeat):
    network = MultiLayerNeuralNetwork(
        train_x[0].size, train_y[0].size, [20, 15, 10])

    train_results = network.train(1000, 0.5, 0.7, train_x, train_y)
    train_error += train_results[0]
    iteration += train_results[1]

    test_results = network.test(test_x, test_y)
    test_error += test_results[0]
    predicted_true += test_results[1]
    predicted_false += test_results[2]

iteration_average = iteration/number_of_repeat
error_average = train_error/number_of_repeat
test_error_average = test_error/number_of_repeat
predicted_true_average = predicted_true/number_of_repeat
predicted_false_average = predicted_false/number_of_repeat

print(iteration_average)
print(error_average)
print(test_error_average)
print(predicted_true_average)
print(predicted_false_average)

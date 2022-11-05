import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return (1/(1+np.exp(-x)))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s*(1-s)


class MultiLayerNeuralNetwork:
    def __init__(self, input_dim, output_dim, layer_array):
        # add output layer according to output length
        self.layer_array = np.append(layer_array, [output_dim]).tolist()
        self.layer_count = len(self.layer_array)

        # initiate weights with random values
        self.weights = self._initiate_weights(
            np.append([input_dim], self.layer_array).tolist())

        # declares previous_weight_change and copy weight_array to previous_weight_change for its first value
        self.previous_weight_change = np.zeros(self.layer_count).tolist()
        for idx, val in enumerate(self.previous_weight_change):
            self.previous_weight_change[idx] = np.empty_like(
                self.weights[idx])
            self.previous_weight_change[idx][:] = self.weights[idx]

    def _initiate_weights(self, array):
        weight_array = np.zeros(self.layer_count).tolist()
        for idx, val in enumerate(array):
            if(idx != 0):
                weight_array[idx-1] = np.random.uniform(low=-0.5,
                                                        high=0.5, size=(array[idx], array[idx-1]+1))
        return weight_array

    # forward() multiplies x and w values
    # returns input, v, y values for each neuron

    def _train_forward(self, input, input_target):
        # list for input, v, y values
        temp_v_array = np.zeros(self.layer_count).tolist()
        temp_y_array = np.zeros(self.layer_count).tolist()

        layer_input = np.zeros(self.layer_count).tolist()

        # define a variable that takes the value of previous neuron out
        # previous neuron out is input of next neuron
        # for the layer inputs are input of the network
        y = input
        for idx, val in enumerate(self.weights):
            y = np.append(y, [1])       # adding bias
            layer_input[idx] = y

            v = np.matmul(val, y)
            y = sigmoid(v)

            temp_v_array[idx] = v
            temp_y_array[idx] = y

            if(len(self.weights) == idx+1):  # calculate error and out of network
                output = y
                error = input_target - y

        return temp_v_array, temp_y_array, error, layer_input, output

    def _back_propogation(self, e, v_array):
        # e is error vector (e1, e2)
        # create a weight list without elemets that correspond to bias
        weight_array_wo_bias = np.zeros(self.layer_count).tolist()
        for idx, val in enumerate(self.weights):
            weight_array_wo_bias[idx] = np.delete(val, val.shape[1]-1, 1)

        # decleare local gradient list
        local_gradient = np.zeros(self.layer_count).tolist()

        # calculate the first element of local gradient list
        local_gradient[self.layer_count-1] = e * \
            sigmoid_derivative(v_array[self.layer_count-1])

        # calculate the other elements of local gradient list
        for idx in reversed(range(self.layer_count-1)):

            local_gradient[idx] = np.matmul(np.transpose(
                weight_array_wo_bias[idx+1]), local_gradient[idx+1]) * sigmoid_derivative(v_array[idx])

        return local_gradient

    def _update_weights(self, local_gradient_array, input_layer_list):
        # w(k+1) next values of weights
        temp_weight = np.zeros(self.layer_count).tolist()
        # saving weight_change for momentum
        weight_change = np.zeros(self.layer_count).tolist()

        for idx, i in enumerate(self.weights):
            # fixing the shapes
            local_gradient_array[idx].shape = (
                len(local_gradient_array[idx]), 1)
            input_layer_list[idx].shape = (1, len(input_layer_list[idx]))

            weight_change[idx] = (np.matmul(local_gradient_array[idx], input_layer_list[idx])
                                  * self.learning_coef) + (self.previous_weight_change[idx] * self.momentum_coef)
            temp_weight[idx] = i + weight_change[idx]

        return temp_weight, weight_change

    def _multi_layer_method(self, input, output):
        train_forward_result = self._train_forward(input, output)
        v_array = train_forward_result[0]
        y_array = train_forward_result[1]

        # calculating numerical value of error vector
        # train_forward_result[2] return error vector
        error_vector = train_forward_result[2]
        err = np.matmul(error_vector, np.transpose(error_vector)) / 2

        input_layer_list = train_forward_result[3]
        result = train_forward_result[4]

        local_gradient_array = self._back_propogation(error_vector, v_array)

        update_weights_result = self._update_weights(
            local_gradient_array, input_layer_list)
        updated_weights = update_weights_result[0]
        weight_change = update_weights_result[1]

        return updated_weights, weight_change, result, err

    def train(self, epoch, learning_coef, momentum_coef, input, output):
        self.epoch = epoch
        self.learning_coef = learning_coef
        self.momentum_coef = momentum_coef
        self.input = input
        self.output = output

        index = np.arange(len(input))  # for random inputs for each iteration
        error = 0
        iteration = 0
        for i in range(epoch):
            np.random.shuffle(index)
            error = 0
            iteration = iteration + 1

            for idx, j in enumerate(input):
                epoch_result = self._multi_layer_method(
                    input[index[idx]], output[index[idx]])
                self.weights = epoch_result[0]
                self.previous_weight_change = epoch_result[1]
                result = epoch_result[2]
                error += epoch_result[3]

            error = error/len(input)
            if(error < 0.0001):
                break

        return error, iteration

    def _test_forward(self, input, target):
        y = input
        for idx, val in enumerate(self.weights):
            y = np.append(y, [1])       # adding bias
            v = np.matmul(val, y)
            y = sigmoid(v)
            if(len(self.weights) == idx+1):  # calculate error and out of network
                output = y
                err_vector = target - y
                err = np.matmul(err_vector, np.transpose(err_vector)) / 2
        return err_vector, err, output

    def test(self, test_x, test_y):
        predicted_true = 0
        predicted_false = 0
        error = 0
        for idx, val in enumerate(test_x):
            test_forward_result = self._test_forward(test_x[idx], test_y[idx])
            error_vector = test_forward_result[0]
            error += test_forward_result[1]
            output = test_forward_result[2]

            if(np.argmax(output) == np.argmax(test_y[idx])):
                predicted_true += 1
            else:
                predicted_false += 1

        error = error/len(test_x)

        return error, predicted_true, predicted_false

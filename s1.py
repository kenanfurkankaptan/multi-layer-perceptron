import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt


# training and test lists
train_x = np.zeros(20).tolist()
train_y = np.zeros(20).tolist()

test_x = np.zeros(12).tolist()
test_y = np.zeros(12).tolist()

# load array
train_x = loadtxt('train_x.csv', delimiter=',')
train_y = loadtxt('train_y.csv', delimiter=',')
test_x = loadtxt('test_x.csv', delimiter=',')
test_y = loadtxt('test_y.csv', delimiter=',')


def sigmoid(x):
    return (1/(1+np.exp(-x)))

def sigmoid_derivative(x):
    s = sigmoid(x) 
    return s*(1-s)



def multi_layer_NN(e, learning_coef, momentum_coef, input, target, layer_array):

    layer_count = len(layer_array)
    input_length = len(input[0])
    epoch = e

    temp_layer_array = np.append([input_length], layer_array).tolist()


    def initiate_weights(array):
        temp_list = np.zeros(layer_count)
        temp_list = temp_list.tolist()
        for idx, val in enumerate(array):
            if(idx!=0):
                temp_list[idx-1] = np.random.uniform(low = -0.5, high = 0.5, size=(array[idx], array[idx-1]+1))
        return temp_list


    # calculates and returns updated_weights, weight_change, result, error for one input vector
    def multi_layer_method(weight, learning_rate, input, input_target, previous_weight_change, momentum_rate):


        # forward() multiplies x and w values 
        # returns input, v, y values for each neuron
        def forward():

            # list for input, v, y values
            temp_v_array = np.zeros(layer_count)
            temp_y_array = np.zeros(layer_count)
            temp_v_array = temp_v_array.tolist()
            temp_y_array = temp_y_array.tolist()

            layer_input = np.zeros(layer_count)
            layer_input = layer_input.tolist()


            # define a variable that takes the value of previous neuron out
            # previous neuron out is input of next neuron
            # for the layer inputs are input of the network
            y = input

            for idx, val in enumerate(weight):

                y = np.append(y, [1])       # adding bias 
                layer_input[idx] = y

                v = np.matmul(val, y)
                y = sigmoid(v)

                temp_v_array[idx] = v
                temp_y_array[idx] = y

                if(len(weight) == idx+1):    #calculate error and out of network
                    output = y
                    error = input_target - y

            return temp_v_array, temp_y_array, error, layer_input, output


        def back_propogation(e):    # e is error vector (e1, e2)

            # create a weight list without elemets that correspond to bias  
            weight_array_wo_bias = np.zeros(layer_count).tolist()
            for idx, val in enumerate(weight):
                weight_array_wo_bias[idx] = np.delete(val, val.shape[1]-1, 1)

            local_gradient = np.zeros(layer_count).tolist()     # decleare local gradient list

            # calculate the first element of local gradient list
            local_gradient[layer_count-1] =  e * sigmoid_derivative(v_array[layer_count-1])

            for idx in reversed(range(layer_count-1)):      # calculate the other elements of local gradient list

                local_gradient[idx] = np.matmul(np.transpose(weight_array_wo_bias[idx+1]), local_gradient[idx+1]) * sigmoid_derivative(v_array[idx])

            return local_gradient


        def update_weights():
            
            temp_weight = np.zeros(layer_count).tolist()        # w(k+1) next values of weights
            weight_change = np.zeros(layer_count).tolist()      # saving weight_change for momentum

            for idx, i in enumerate(weight):
                
                # fixing the shapes
                local_gradient_array[idx].shape = (len(local_gradient_array[idx]), 1)
                input_layer_list[idx].shape = (1, len(input_layer_list[idx]))

                weight_change[idx] = (np.matmul(local_gradient_array[idx], input_layer_list[idx]) * learning_rate) + (previous_weight_change[idx] * momentum_rate)
                temp_weight[idx] = i + weight_change[idx]

            return temp_weight, weight_change


        temp_var1 = forward()

        v_array = temp_var1[0]
        y_array = temp_var1[1]

        # calculating numerical value of error vector
        # temp_var1[2] return error vector
        error_vector = temp_var1[2]
        err = np.matmul(error_vector, np.transpose(error_vector)) / 2

        input_layer_list = temp_var1[3]
        result = temp_var1[4]

        local_gradient_array = back_propogation(error_vector)


        temp_var2 = update_weights()
        updated_weights = temp_var2[0]
        weight_change = temp_var2[1]

        return updated_weights, weight_change, result, err





    # declares previous_weight_change and copy weight_array to previous_weight_change for its first value
    weight_array = initiate_weights(temp_layer_array)
    previous_weight_change = np.zeros(layer_count).tolist()
    for idx, val in enumerate(previous_weight_change):
        previous_weight_change[idx] = np.empty_like(weight_array[idx])
        previous_weight_change[idx][:] = weight_array[idx]



    index = np.arange(len(input))       #for random inputs for each iteration
    error = 0
    iteration = 0
    for i in range(epoch):

        np.random.shuffle(index)
        error = 0
        iteration = iteration + 1

        for idx, j in enumerate(input):
            temp1 = multi_layer_method(weight_array, learning_coef, input[index[idx]], target[index[idx]], previous_weight_change, momentum_coef)
            weight_array = temp1[0]
            previous_weight_change = temp1[1]
            result = temp1[2]
            error += temp1[3]

        error = error/len(input)
        if(error < 0.0001):
            break

    return error, iteration, weight_array


def test(weight):

    def forward(input, target, weight):
        y = input
        for idx, val in enumerate(weight):
            y = np.append(y, [1])       # adding bias 
            v = np.matmul(val, y)
            y = sigmoid(v)
            if(len(weight) == idx+1):    #calculate error and out of network
                output = y
                err_vector = target - y
                err = np.matmul(err_vector, np.transpose(err_vector)) / 2
        return err_vector, err, output

    predicted_true = 0
    predicted_false = 0
    error = 0
    for idx, val in enumerate(test_x):
        temp = forward(test_x[idx], test_y[idx], weight)
        error_vector = temp[0]
        error += temp[1]
        output = temp[2]

        if(np.argmax(output) == np.argmax(test_y[idx])):
            predicted_true += 1
        else:
            predicted_false += 1

    error = error/12


    return error, predicted_true, predicted_false



train_error = 0
iteration = 0
test_error = 0
predicted_true = 0
predicted_false = 0

for i in range(20):
    temp1 = multi_layer_NN(1000, 0.5, 0.7, train_x, train_y, [10, 10, 10, 4])
    train_error += temp1[0]
    iteration += temp1[1]
    weights =  temp1[2]

    temp2 = test(weights)
    test_error += temp2[0]
    predicted_true += temp2[1]
    predicted_false += temp2[2]


iteration_average = iteration/20
error_average = train_error/20
test_error_average = test_error/20
predicted_true_average = predicted_true/20
predicted_false_average = predicted_false/20

print(iteration_average)
print(error_average)
print(test_error_average)
print(predicted_true_average)
print(predicted_false_average)
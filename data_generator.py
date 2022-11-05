import matplotlib.pyplot as plt
import numpy as np
from numpy import savetxt


def data_generator(example_x, example_y, train_len, test_len):

    example_len = len(example_x)
    array_len = train_len + test_len

    data_x = np.zeros(array_len).tolist()
    data_y = np.zeros(array_len).tolist()

    boolean = False
    for i in range(array_len):
        if(i % example_len * 4 == 0):
            boolean = not boolean

        rand_index = np.random.randint(0, 50)

        temp_x = np.empty_like(example_x[0])
        temp_x[:] = example_x[i % len(example_x)]
        if(boolean):
            # flips one element of the array
            temp_x[int(rand_index % 5), int(rand_index / 5)] = np.where(
                temp_x[int(rand_index % 5), int(rand_index / 5)] == 0.9,
                0.1, 0.9)
        else:
            # adds noise to all array
            noisy = temp_x + 0.2 * np.random.rand(5, 10)
            temp_x = noisy/noisy.max()

        data_x[i] = temp_x
        data_y[i] = example_y[i % len(example_x)]

    # reshape the data matrices
    #  5x10  --->>  50x1
    # shape 5x10 matrices to 50x1 vectors
    for idx, val in enumerate(data_x):
        temp_data = np.empty_like(val)
        temp_data[:] = val
        data_x[idx] = np.concatenate(
            (temp_data[0], temp_data[1], temp_data[2], temp_data[3], temp_data[4]), axis=None)

    # create test and train list and divide the data
    train_x = np.zeros(train_len).tolist()
    train_y = np.zeros(train_len).tolist()

    test_x = np.zeros(test_len).tolist()
    test_y = np.zeros(test_len).tolist()

    for idx, val in enumerate(data_x):
        if(idx < train_len):
            train_x[idx] = data_x[idx]
            train_y[idx] = data_y[idx]
        else:
            test_x[idx-train_len] = data_x[idx]
            test_y[idx-train_len] = data_y[idx]

    # save to csv file
    savetxt('train_x.csv', train_x, delimiter=',')
    savetxt('train_y.csv', train_y, delimiter=',')
    savetxt('test_x.csv', test_x, delimiter=',')
    savetxt('test_y.csv', test_y, delimiter=',')

    return

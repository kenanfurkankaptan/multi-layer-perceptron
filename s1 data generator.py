import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
from numpy import savetxt



data_x = np.zeros(32).tolist()
data_y = np.zeros(32).tolist()


# O
data_y[0] = np.array([0.9, 0.1, 0.1, 0.1])
data_x[0] = np.array([   
    np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1]),
    np.array([0.1, 0.1, 0.1, 0.9, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1]),
    np.array([0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1]),
    np.array([0.1, 0.1, 0.1, 0.9, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1]),
    np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1])
])

# L
data_y[1] = np.array([0.1, 0.9, 0.1, 0.1])
data_x[1] = np.array([   
    np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    np.array([0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
])

# N
data_y[2] = np.array([0.1, 0.1, 0.9, 0.1])
data_x[2] = np.array([   
    np.array([0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1]),
    np.array([0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1]),
    np.array([0.1, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.1, 0.1]),
    np.array([0.1, 0.1, 0.9, 0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.1]),
    np.array([0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1])
])

# E
data_y[3] = np.array([0.1, 0.1, 0.1, 0.9])
data_x[3] = np.array([   
    np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1]),
    np.array([0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1]),
    np.array([0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1])
])


# plt.imshow(data_x[0], cmap='gray')
# plt.show()
# plt.imshow(data_x[1], cmap='gray')
# plt.show()
# plt.imshow(data_x[2], cmap='gray')
# plt.show()
# plt.imshow(data_x[3], cmap='gray')
# plt.show()



boolean = False
for i in range(4, len(data_x)):
    
    if(i%4 == 0):
        boolean = not boolean

    temp_val= data_y[i%4]
    data_y[i] = temp_val
    
    # array assign by value 
    temp = np.empty_like(data_x[0]) 
    temp[:] = data_x[i%4]

    rand_index = np.random.randint(0, 50)

    if(boolean):
        temp[int(rand_index % 5), int(rand_index / 5)] = np.where(
 temp[int(rand_index % 5), int(rand_index / 5)] == 0.9,
 0.1, 0.9)

    else:
        noisy = temp + 0.2 * np.random.rand(5, 10)
        temp = noisy/noisy.max()


    data_x[i] = temp

    plt.imshow(data_x[i], cmap='gray')
    plt.show()




# reshape the data matrices
#  5x10  --->>  50x1
# shape 5x10 matrices to 50x1 vectors
for idx, val in enumerate(data_x):
    temp = np.empty_like(val) 
    temp[:] = val
    data_x[idx] = np.concatenate((temp[0], temp[1], temp[2], temp[3], temp[4]), axis=None)



# create test and train list and divide the data
train_x = np.zeros(20).tolist()
train_y = np.zeros(20).tolist()

test_x = np.zeros(12).tolist()
test_y = np.zeros(12).tolist()

for idx, val in enumerate(data_x):
    if(idx<20):
        train_x[idx] = data_x[idx]
        train_y[idx] = data_y[idx]
    else:
        test_x[idx-20] = data_x[idx]
        test_y[idx-20] = data_y[idx]




# save to csv file
savetxt('train_x.csv', train_x, delimiter=',')
savetxt('train_y.csv', train_y, delimiter=',')
savetxt('test_x.csv', test_x, delimiter=',')
savetxt('test_y.csv', test_y, delimiter=',')


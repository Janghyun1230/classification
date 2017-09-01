import numpy as np
import matplotlib.pyplot as plt


def moons(center1= (0,0), center2=(1,0.5), sample_per_class= 200, print_data = False, iter=1):
    data_train = np.zeros((sample_per_class*iter*2, 2))
    label_train = np.zeros(sample_per_class*iter*2)
    for it in range(iter):
        for i in range(sample_per_class):
            theta = np.random.uniform(0, np.pi, 1)
            e = np.random.uniform(-0.1,0.1,2)
            data_train[i+it*sample_per_class*2,:] = [np.cos(theta)+center1[0]+it*2+e[0], np.sin(theta)+center1[1] + e[1]]
            label_train[i+it*sample_per_class*2] = 0
        for i in range(sample_per_class, sample_per_class*2):
            theta = np.random.uniform(-np.pi, 0, 1)
            e = np.random.uniform(-0.1, 0.1, 2)
            data_train[i+it*sample_per_class*2,:] = [np.cos(theta)+center2[0]+it*2+e[0], np.sin(theta)+center2[1] + e[1]]
            label_train[i+it*sample_per_class*2] = 1

    if print_data:
        plt.scatter(data_train[:,0], data_train[:,1], c = label_train, s = 2)
        plt.show()
    x_range = [-1.5, 2.5*iter]
    y_range = [-0.75, 1.25]
    return data_train, label_train.reshape(-1,1), x_range, y_range


def circle(sample_per_class= 200, print_data = False, iter=1):
    data_train = np.zeros((sample_per_class*iter*2, 2))
    label_train = np.zeros(sample_per_class*iter*2)
    for it in range(iter):
        for i in range(sample_per_class):
            theta = np.random.uniform(-np.pi, np.pi, 1)
            e = np.random.uniform(-0.1,0.1,2)
            data_train[i+it*sample_per_class*2,:] = [np.sin(theta)*((it+1)*2-1)+e[0], np.cos(theta)*((it+1)*2-1) + e[1]]
            label_train[i+it*sample_per_class*2] = 0
        for i in range(sample_per_class, sample_per_class*2):
            theta = np.random.uniform(-np.pi, np.pi, 1)
            e = np.random.uniform(-0.1, 0.1, 2)
            data_train[i+it*sample_per_class*2,:] = [np.sin(theta)*(it+1)*2+e[0], np.cos(theta)*(it+1)*2 + e[1]]
            label_train[i+it*sample_per_class*2] = 1

    if print_data:
        plt.scatter(data_train[:,0], data_train[:,1], c = label_train, s = 2)
        plt.show()
    x_range = [-2.5*iter, 2.5*iter]
    y_range = [-2.5*iter, 2.5*iter]
    return data_train, label_train.reshape(-1,1), x_range, y_range

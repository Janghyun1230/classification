import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from network import *
from optim import *
import time
from classification_dataset import moons, circle

tf.reset_default_graph()

batch_size = 500
epoch = 400
print_per_epoch = 10
lr_init = 4e-3
lr_decay_epoch = 200

layers = [('fc1', {'n_out': 10, 'act': relu, 'batch': True}),
          ('fc2', {'n_out': 10, 'act': relu, 'batch': True}),
          ('fc7', {'n_out': 1, 'act': sigmoid})
          ]

lr = tf.placeholder(tf.float32)
net_optimizer = optimizer("adam", lr=lr, beta1=0.5, beta2=0.999)


# data set
data, label, x_range, y_range = moons(sample_per_class= 250, iter=3)


# graph
input = tf.placeholder(dtype = tf.float32, shape=(batch_size, 2))
output = tf.placeholder(dtype = tf.float32, shape=(batch_size,1))
is_training = tf.placeholder(dtype= tf.bool)

prob, tensors, vars, moving_avg = network('classification', input, layers, is_training=is_training)
loss = tf.reduce_mean(-output*tf.log(prob) - (1-output)*tf.log(1-prob))

with tf.control_dependencies(moving_avg):
    train_step = net_optimizer.minimize(loss, var_list=vars)

saver = tf.train.Saver()


# train and print
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    data_size = data.shape[0]
    iter_per_epoch = data_size//batch_size
    print("\n training start...")
    print("total iteration : ", iter_per_epoch*epoch)
    tic = time.time()

    print_cell = int(np.ceil(np.sqrt(epoch/print_per_epoch)))
    plt.figure(figsize=(20,20))
    loss_hist = []

    for j in range(epoch):
        # shuffle data set
        arr = np.arange(data_size)
        np.random.shuffle(arr)

        # train step
        for i in range(iter_per_epoch):
            _, loss_cur = sess.run([train_step,loss],
                                   feed_dict= {input: data[arr[batch_size*i: batch_size*(i+1)]],
                                               output: label[arr[batch_size*i: batch_size*(i+1)]],
                                               is_training: True,lr : lr_init})
            loss_hist.append(loss_cur)

        # print loss and accuracy
        print("epoch : %d" % (j+1))
        print("loss : %0.3f" % loss_cur)
        y_out = np.rint(sess.run(prob, feed_dict={input: data[arr[: batch_size]], is_training: False}))
        acc = np.mean(y_out == label[arr[: batch_size]])
        print("accuracy : ", acc)

        # draw prob = 0.5 line
        if j % print_per_epoch == 0:
            plt.subplot(print_cell, print_cell, j//print_per_epoch+1)
            x1 = np.linspace(x_range[0], x_range[1], batch_size)
            y1 = np.linspace(y_range[0], y_range[1], batch_size)
            X1, Y1 = np.meshgrid(x1, y1)
            Z = np.zeros((batch_size, batch_size))
            for i in range(batch_size):
                Z[i, :] = np.rint(
                    sess.run(prob, feed_dict={input: np.hstack([X1[i].reshape(-1, 1), Y1[i].reshape(-1, 1)]),
                                              is_training: False})).ravel()

            plt.scatter(data[:, 0], data[:, 1], c=label, s=2)
            CS = plt.contour(X1, Y1, Z, 1, colors='g')
            plt.clabel(CS, inline=1, fontsize=10)
            plt.axis('off')
            plt.title("epoch %d"%(j+1))

        # decay learning rate
        if j%lr_decay_epoch == 0 and j!=0:
            lr_init /= 2.0
            print("learning rate is decayed : %f "%lr_init)
            plt.title("lr decayed")

    # print training time
    print("training end")
    toc = time.time()
    print("time : %f"%(toc-tic))

    # plot loss
    plt.subplot(print_cell, print_cell, print_cell**2)
    plt.plot(loss_hist)
    plt.title("time : %0.2f"%(toc-tic))
    plt.ylabel("loss")
    plt.show()
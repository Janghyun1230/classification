# classification
classsify 2d data set by deep neural network

### classification_dataset.py
>functions which generate datasets. now has circle, moons

### network.py
>activation functions and network function which draw tensorflow layers by input dictionary.
>now it can draw fullyconnected, conv2D, transpose conv2D with reshape, batch nomalization. I'll update concat and residual.

### optim.py
>optimizer function which return tensorflow optimizer.

### classification.py
>you can input hyperparameters. e.g. batch_size, epoch, learning rate...

# output
#### double circle
![Alt text](/output/double_circle_2hidden_batch.png?raw=true "classify double circle")

#### triple moons
![Alt text](/output/triple_moons_2hidden_batch.png )


# Summary
- Batch norm is good (In tf.contrib.layers.batchnorm default decay is 0.999, but this not works well. I use 0.9 for decay.)
- The deep is better than the fat network
- Decaying learning rate is good (by factor 2 i.e. /=2)
- Small batch size is efficient(fast), but sometimes it is hard to train. 

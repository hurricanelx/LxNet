# LxNet
A very simple CNN with Numpy

## Layers

contain Maxpool,ReLU,FC,Conv,AvgPool and Softmax
All these layers are implemented by Numpy with loops


## Solver

The Solver class help to train the model and to adjust each layer with the gradient 

## dataset

I use cifar-10 as an example. CIFAR:http://www.cs.toronto.edu/~kriz/cifar.html</br>
dataset should be put into the dataset directory

## test

A very simple example is written in test.py</br></br>
You may find that the model can not fit the data very well.</br></br>
It may due to the very shallow structure.

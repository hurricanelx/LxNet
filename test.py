from data_utils import get_CIFAR10_data
import numpy as np
from Layers.conv_layer import conv
from Layers.ReLU import ReLU
from Layers.FC_Layer import FC_Layer
from Layers.Softmax import Softmax
from Layers.MaxPooling import MaxPooling
from LxNet import Solver

data = get_CIFAR10_data()
x_train = data['x_train']
y_train = data['y_train']
x_val = data['x_val']
y_val = data['y_val']
x_test = data['x_test']
y_test = data['y_test']

N = x_train.shape[0]
epoch = 10
batch_size = 64
stride = 1
padding = 1
kernel_size = 3
lr = 4e-1

conv1_1 = conv((3, 32, 32), 32, kernel_size, stride, padding)
ReLu1_1 = ReLU()
MaxPooling1 = MaxPooling()
conv2_1 = conv((32, 16, 16), 32, kernel_size, stride, padding)
ReLu2_1 = ReLU()
MaxPooling2 = MaxPooling()
FC1 = FC_Layer(32 * 8 * 8, 256)
FC2 = FC_Layer(256, 10)
Softmax_classifier = Softmax()

model = [conv1_1, ReLu1_1, MaxPooling1, conv2_1, ReLu2_1, MaxPooling2, FC1, FC2]
solver = Solver.Solver(model)

solver.train(x_train, y_train, epoch, batch_size, lr, 1)

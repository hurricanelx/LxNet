import numpy as np
from Layers.Softmax import Softmax


class Solver:
    def __init__(self, model):
        self.model = model

    def forward(self, x, y):
        temp = x
        soft = Softmax()
        for layer in self.model:
            out = layer.forward(temp)
            temp = out
        loss, d1 = soft.loss(out, y)
        return loss, d1

    def train(self, x_train, y_train, epoch, batch_size, lr, num_print=2):
        N = x_train.shape[0]
        for i in range(epoch):
            num_itration = N // batch_size + 1
            num_correct = 0
            num_test = 0
            for j in range(num_itration):
                temp_x = x_train[j * batch_size:(j + 1) * batch_size] / 255
                temp_y = y_train[j * batch_size:(j + 1) * batch_size]
                loss, d1 = self.forward(temp_x, temp_y)

                if j % num_print == 0:
                    print(loss)

                dtemp = d1
                for layers in self.model[::-1]:
                    dout = layers.optim(dtemp, lr)
                    dtemp = dout

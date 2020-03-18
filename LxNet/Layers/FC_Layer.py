import numpy as np


class FC_Layer:
    def __init__(self, in_channel, out_channel):
        self.D = in_channel
        self.out_channel = out_channel
        self.w = 1e-3 * np.random.randn(self.D, self.out_channel)
        self.b = np.zeros(self.out_channel)

    def forward(self, x):
        self.x = x
        self.new_x = np.reshape(x, (x.shape[0], -1))
        temp_x = self.new_x.dot(self.w)
        out = temp_x + self.b
        return out

    def backward(self, dout):
        db = np.sum(dout, axis=0)
        dw = self.new_x.T.dot(dout)
        dx = dout.dot(self.w.T)
        dx = np.reshape(dx, self.x.shape)
        return dx, dw, db

    def optim(self, dout, lr=1e-5):
        dx, dw, db = self.backward(dout)
        self.w -= lr * dw
        self.b -= lr * db
        return dx

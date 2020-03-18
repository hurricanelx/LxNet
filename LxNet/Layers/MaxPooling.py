import numpy as np


class MaxPooling:
    def __init__(self, pool=(2, 2), stride=2):
        if len(pool) == 2:
            self.pool_height = pool[0]
            self.pool_width = pool[1]
        elif len(pool) == 1:
            self.pool_height = self.pool_width = pool
        self.stride = stride

    def forward(self, x):
        self.x = x
        self.N, self.C, self.H, self.W = x.shape
        self.H_new = 1 + (self.H - self.pool_height) // self.stride
        self.W_new = 1 + (self.W - self.pool_width) // self.stride
        out = np.zeros((self.N, self.C, self.H_new, self.W_new))
        for i in range(self.H_new):
            for j in range(self.W_new):
                out[:, :, i, j] = np.max(
                    self.x[:, :,
                    i * self.stride:i * self.stride + self.pool_height,
                    j * self.stride:j * self.stride + self.pool_width], axis=(2, 3))
        return out

    def backward(self, dout):
        dx = np.zeros_like(self.x)
        for i in range(self.H_new):
            for j in range(self.W_new):
                max_num = np.max(
                    self.x[:, :,
                    i * self.stride:i * self.stride + self.pool_height,
                    j * self.stride:j * self.stride + self.pool_width], axis=(2, 3))
                mask = (self.x[:, :,
                        i * self.stride:i * self.stride + self.pool_height,
                        j * self.stride:j * self.stride + self.pool_width] == max_num[:, :, None, None])
                dx[:, :,
                i * self.stride:i * self.stride + self.pool_height,
                j * self.stride:j * self.stride + self.pool_width] += (dout[:, :, i, j])[:, :, None, None] * mask
        return dx

    def optim(self, dout, lr):
        dx = self.backward(dout)
        return dx
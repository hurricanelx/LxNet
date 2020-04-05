import numpy as np


class AverPooling:
    """
    The average pooling layer, which output is the average value of the small parts of inputs
    """
    def __init__(self, pool=(2, 2), stride=2):
        """
        :param pool: A tuple or a list
        :param stride: A integer
        """
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
                out[:, :, i, j] = np.mean(
                    self.x[:, :,
                    i * self.stride:i * self.stride + self.pool_height,
                    j * self.stride:j * self.stride + self.pool_width], axis=(2, 3))
        return out

    def backward(self, dout):
        dx = np.zeros_like(self.x, dtype=np.float)
        num = float(self.H_new * self.W_new)
        for i in range(self.H_new):
            for j in range(self.W_new):
                dx[:, :,
                i * self.stride:i * self.stride + self.pool_height,
                j * self.stride:j * self.stride + self.pool_width] += dout[:, :, i, j] / num
        return dx


if __name__ == '__main__':
    kind = np.arange(16)
    kind = np.reshape(kind, (2, 1, 2, 4))
    av = AverPooling()
    out = av.forward(kind)
    print(out)
    dout = np.ones_like(out)
    temp = av.backward(dout)
    print(temp)

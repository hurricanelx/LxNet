import numpy as np


class ReLU:
    def forward(self, x):
        self.x = x
        out = np.maximum(self.x, 0)
        return out

    def backward(self, dout):
        dx = dout * (self.x >= 0)
        # hang, lie = np.where(self.x <= 0)
        # dx = dout
        # dx[hang, lie] = 0
        return dx

    def optim(self, dout, lr):
        dx = self.backward(dout)
        return dx

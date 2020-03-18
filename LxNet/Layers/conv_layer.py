import numpy as np


class conv:
    def __init__(self, in_channel, out_channel, kernel_size, stride, pad=0):
        self.C, self.H, self.W = in_channel
        self.F = out_channel
        if type(kernel_size) == tuple:
            self.HH = kernel_size[0]
            self.WW = kernel_size[1]
        elif type(kernel_size) == int:
            self.HH = self.WW = kernel_size
        self.stride = stride
        self.pad = pad

        self.H_new = 1 + (self.H + 2 * pad - self.HH) // stride
        self.W_new = 1 + (self.W + 2 * pad - self.WW) // stride
        self.w = 1e-3 * np.random.randn(self.F, self.C, self.HH, self.WW)
        self.b = np.zeros(self.F)

    def forward(self, x):
        self.x = x
        self.N = x.shape[0]
        N, F, HH, WW, H_new, W_new = self.N, self.F, self.HH, self.WW, self.H_new, self.W_new
        x, w, b = self.x, self.w, self.b
        pad = self.pad
        stride = self.stride
        out = np.zeros((N, F, H_new, W_new))
        for z in range(N):
            pad_x = np.pad(x[z], ((0, 0), (pad, pad), (pad, pad)), 'constant')
            for y in range(F):
                w_y = w[y]
                for i in range(H_new):
                    for j in range(W_new):
                        temp_x = pad_x[:, i * stride:i * stride + HH, j * stride:j * stride + WW]
                        out[z, y, i, j] = np.sum(np.multiply(temp_x, w_y))
        b = np.reshape(b, (F, 1, 1))
        out += b
        return out

    def backward(self, dout):
        N, F, HH, WW, H_new, W_new = self.N, self.F, self.HH, self.WW, self.H_new, self.W_new
        x, w, b = self.x, self.w, self.b
        pad = self.pad
        stride = self.stride
        dx = np.zeros_like(x)
        dw = np.zeros_like(w)
        db = np.sum(dout, axis=(0, 2, 3))

        for z in range(N):
            pad_x = np.pad(x[z], ((0, 0), (pad, pad), (pad, pad)), 'constant')
            dpadx = np.zeros_like(pad_x)
            for y in range(F):
                for i in range(H_new):
                    for j in range(W_new):
                        temp_x = pad_x[:, i * stride:i * stride + HH, j * stride:j * stride + WW]
                        dw[y] += dout[z, y, i, j] * temp_x
                        dpadx[:, i * stride:i * stride + HH, j * stride:j * stride + WW] += dout[z, y, i, j] * w[y]
            dx[z] = dpadx[:, pad:-pad, pad:-pad]
        return dx, dw, db

    def optim(self, dout, lr=1e-5):
        dx, dw, db = self.backward(dout)
        self.W -= lr*dw
        self.b -= lr*db
        return dx
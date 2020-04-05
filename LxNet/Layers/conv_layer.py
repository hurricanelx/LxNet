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

    def forward_fft(self, x):
        self.x = x
        self.N = x.shape[0]
        N, C, F, H, HH, WW, H_new, W_new = self.N, self.C, self.F, self.H, self.HH, self.WW, self.H_new, self.W_new
        x, w, b = self.x, self.w, self.b
        pad = self.pad
        stride = self.stride
        out = np.zeros((N, F, H_new, W_new))
        YN = H + HH - 1
        ccc = HH - 1 - pad
        FFT_N = 2 ** (int(np.log2(YN)) + 1)
        self.fft_x = []
        for z in range(N):
            fft_x_n = []
            for i in range(C):
                fft_x_n.append(np.fft.fft2(x[z][i], (FFT_N, FFT_N)))
            self.fft_w = []
            for y in range(F):
                fft_w_F = []
                for i in range(C):
                    fft_w = np.fft.fft2(w[y][i], (FFT_N, FFT_N))
                    fft_w_F.append(fft_w)
                    fft_re = fft_w * fft_x_n[i]
                    re = np.fft.ifft2(fft_re).real[ccc:(YN - ccc):stride, ccc:(YN - ccc):stride]
                    out[z][y] += re
                self.fft_w.append(fft_w_F)
            self.fft_x.append(fft_x_n)
        b = np.reshape(b, (F, 1, 1))
        out += b

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

    def backward_fft(self, dout):
        N, C, F, H, HH, WW, H_new, W_new = self.N, self.C, self.F, self.H, self.HH, self.WW, self.H_new, self.W_new
        W = self.W
        x, w, b = self.x, self.w, self.b
        fft_w = self.fft_w
        fft_x = self.fft_x
        pad = self.pad
        stride = self.stride
        dx = np.zeros_like(x)
        dw = np.zeros_like(w)
        db = np.sum(dout, axis=(0, 2, 3))

        YN = H + HH - 1
        ccc = HH - 1 - pad
        FFT_N = 2 ** (int(np.log2(YN)) + 1)

        for z in range(N):
            for f in range(F):
                ttt = np.zeros((H_new * stride, W_new * stride))
                ttt[::stride][::stride] = dout[z][f]
                temp_dout = np.pad(ttt, ((ccc, ccc), (ccc, ccc)), 'constant')
                fft_dout = np.fft.ifft2(temp_dout, (FFT_N, FFT_N))
                for c in range(C):
                    temp_dx = fft_dout * fft_w[z][c]
                    temp_dw = fft_dout * fft_x[f][c]
                    temp_dx = np.fft.fft2(temp_dx).real[:H][:W]
                    temp_dw = np.fft.fft2(temp_dw).real[:HH][:WW]
                    dx[z][c] = temp_dx
                    dw[f][c] = temp_dw
        return dx, dw, db

    def optim(self, dout, lr=1e-5):
        dx, dw, db = self.backward(dout)
        self.W -= lr * dw
        self.b -= lr * db
        return dx

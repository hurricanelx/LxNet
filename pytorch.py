import torch
import torch.nn as nn
from data_utils import get_CIFAR10_data

data = get_CIFAR10_data()
x_train = torch.Tensor(data['x_train'])
y_train = torch.Tensor(data['y_train']).long()
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
lr = 1e-1


class mymodel(nn.Module):
    def __init__(self):
        super(mymodel, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size, stride, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size, stride, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.model2 = nn.Sequential(
            nn.Linear(32 * 8 * 8, 256),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        out_1 = self.model1(x)
        out_1 = torch.reshape(out_1, (out_1.shape[0], -1))
        out = self.model2(out_1)
        return out


model = mymodel()
optim = torch.optim.SGD(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

for i in range(epoch):
    num_itration = N // batch_size + 1
    num_correct = 0
    num_test = 0
    for j in range(num_itration):
        temp_x = x_train[j * batch_size:(j + 1) * batch_size] / 255
        temp_y = y_train[j * batch_size:(j + 1) * batch_size]

        score = model(temp_x)
        y_pred = torch.argmax(score, 1)
        acc = (y_pred == temp_y).sum()
        num_correct += acc.item()
        num_test += temp_x.shape[0]
        loss = loss_fn(score, temp_y)
        if j % 30 == 0:
            print(num_correct / num_test)
            print(loss.item())
            print('\n')
        optim.zero_grad()
        loss.backward()
        optim.step()

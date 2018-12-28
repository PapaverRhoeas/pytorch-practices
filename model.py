import torch.nn as nn
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter


# input_size = (batch_size, 1, 32, 32)/(Batch_size * C * H * W)
class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5), stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5), stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, self.num_flat_features(x))         # Flatten the data (n, 16, 5, 5) to (n, 16 * 5 * 5）
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def make_model():
    model = LeNet()
    return model


# inp = torch.rand(1, 1, 32, 32)
# writer = SummaryWriter('./log_tbX')
# net = make_model()
# writer.add_graph(model=net, input_to_model=inp)
# writer.close()

# model = LeNet()
# print(model)
# total = sum([param.nelement() for param in model.parameters()])
# print('Number of params: %.2fM' % (total / 1e6))
# params = list(model.parameters())      # 将模型参数转化成list方便输出
# print(params[0].size())  # conv1's .weight
# print('Model\'s conv1.bias {}'.format(model.conv1.bias))
# print('Model\'s conv1.weight {}'.format(model.conv1.weight))


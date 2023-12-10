import torch.nn as nn
import torch.nn.functional as F

class CNN_Mnist(nn.Module):

    def __init__(self):
        super(CNN_Mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        #self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)    
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNN_Cifar10(nn.Module):

    def __init__(self):
        super(CNN_Cifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
    
class CNN_Cifar10_MOON(nn.Module):

    def __init__(self):
        super(CNN_Cifar10_MOON, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), x


class CNN_KWS(nn.Module):

    def __init__(self):
        super(CNN_Kws, self).__init__()
        self.conv1 = nn.Conv2d(1, 28, kernel_size=(10, 4), stride=(1, 1))
        self.conv2 = nn.Conv2d(28, 30, kernel_size=(10, 4), stride=(1, 2))
        #self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1920, 16)
        self.fc2 = nn.Linear(16, 128)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(self.conv2(x))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)    
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
class CNN_EMnist(nn.Module):

    def __init__(self):
        super(CNN_EMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        #self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 47)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)    
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
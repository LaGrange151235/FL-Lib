import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchsummary
import copy
#from pytorchcv.model_provider import get_model as ptcv_get_model

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, size=[]):
        super(ResidualBlock, self).__init__()
        size1 = copy.deepcopy(size)
        if stride == 2:
            size1[0] //= 2
            size1[1] //= 2
        size2 = copy.deepcopy(size1)
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            # nn.BatchNorm2d(outchannel),
            nn.LayerNorm(size1, elementwise_affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(outchannel)
            nn.LayerNorm(size2, elementwise_affine=False)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(outchannel)
                nn.LayerNorm(size1, elementwise_affine=False)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            nn.LayerNorm([32,32], elementwise_affine=False),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1, size=[32,32])
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2, size=[32,32])
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2, size=[16,16])
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2, size=[8,8])
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride, size):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride, size))
            if stride == 2:
                size[0] //= 2
                size[1] //= 2
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # print('forward')
        return out

class ResNet4Cifar(nn.Module):
    def __init__(self, block, num_block, num_classes=10):
        super().__init__()
        self.in_channels = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(16),
            nn.LayerNorm([32, 32], elementwise_affine=False),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 16, 3, 1, [32,32])
        self.conv3_x = self._make_layer(block, 32, 3, 2, [32,32])
        self.conv4_x = self._make_layer(block, 64, 3, 2, [16,16])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride, size):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, size))
            if stride == 2:
                size[0] //= 2
                size[1] //= 2
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

def resnet20(num_classes=10, **kargs):
    """ return a ResNet 20 object
    """
    return ResNet4Cifar(ResidualBlock, [3, 3, 3], num_classes=num_classes)


def ResNet20_Cifar10():
    return ResNet4Cifar(ResidualBlock, [3, 3, 3], num_classes=10)
    # return ptcv_get_model("resnet20_cifar10")

if __name__=="__main__":
    model = ResNet20_Cifar10()
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    all_size = param_size/1024/1024
    print(all_size, "MB")
    print(param_size, param_sum, all_size)
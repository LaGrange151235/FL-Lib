import torch.nn as nn
import torch.nn.functional as F
import math

cfg = {'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}

class VGG(nn.Module):

    def __init__(self, features, num_class=100):
        super(VGG, self).__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )
#    def __init__(self, features, num_class=10):
#        super(VGG, self).__init__()
#        self.features = features
#        self.classifier = nn.Sequential(
#            nn.Linear(512, 512), # nn.Linear(512, 4096); simpler network for fewer classes
#            nn.ReLU(inplace=True),
#            nn.Linear(512, 512), # nn.Linear(4096, 4096)
#            nn.ReLU(inplace=True),
#            nn.Linear(512, num_class) # nn.Linear(4096, num_class)
#        )
#
#        for m in self.modules():
#            if isinstance(m, nn.Conv2d):
#                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                m.weight.data.normal_(0, math.sqrt(2./n))
#                m.bias.data.zero_()

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
        return output

def make_layers(cfg, batch_norm=False):
    layers = []
    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue
        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]
        if batch_norm:
            layers += [nn.BatchNorm2d(l)]
        layers += [nn.ReLU(inplace=True)]
        input_channel = l
    return nn.Sequential(*layers)

def vgg11_bn():
    return VGG(make_layers(cfg['A'], batch_norm=True))

def vgg13_bn():
    return VGG(make_layers(cfg['B'], batch_norm=True))

def VGG16_Cifar10():
    return VGG(make_layers(cfg['D'], batch_norm=True))
#    return VGG(make_layers(cfg['D']))

class VGG11_CIFAR10(nn.Module):
    def __init__(self):
        super(VGG11_CIFAR10, self).__init__()
        cfg=[32, 'M', 64, 'M', 128, 128, 'M', 128, 128, 'M', 128, 128, 'M']
        self.features = self.make_layers(cfg)
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def make_layers(self,cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

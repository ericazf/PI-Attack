import os
import torch
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import models 
import torch.nn as nn 
from torch.hub import load_state_dict_from_url


model_urls = {
    'AlexNet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'VGG11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'VGG13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'VGG16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'VGG19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'VGG11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'VGG13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'VGG16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'VGG19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


def get_alexnet(model_name, num_classes = 10):
    model = models.alexnet(pretrained = True)
    input = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(input, num_classes)
    return model 
    
def get_vgg(model_name, num_classes):
    vgg_dict = {"VGG11":models.vgg11, "VGG13":models.vgg13, "VGG16":models.vgg16, "VGG19":models.vgg19}
    model = vgg_dict[model_name](pretrained=True)
    input = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(input, num_classes)
    return model 

def get_resnet(model_name, num_classes):
    resnet_dict = {"ResNet18":models.resnet18, "ResNet34":models.resnet34,"ResNet50":models.resnet50, "ResNet101":models.resnet101, "ResNet152":models.resnet152}
    model = resnet_dict[model_name](pretrained=True)
    input = model.fc.in_features
    model.fc = torch.nn.Linear(input, num_classes)
    return model 

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        feature = torch.flatten(x, 1)
        y = self.classifier(feature)
        return feature, y 
 
def get_AlexNet(model_name, num_classes, dataset, pretrained = True):
    if pretrained:
        model = AlexNet(model_name, 1000)
        state_dict = load_state_dict_from_url(model_urls[model_name], progress=True)
        model.load_state_dict(state_dict)
        if dataset != "Imagenet":
            input = model.classifier[6].in_features
            model.classifier[6] = torch.nn.Linear(input, num_classes)
    else:
        model = AlexNet(model_name, num_classes)
    return model 
    
class VGG(nn.Module):
    def __init__(self, model_name, num_classes):
        super(VGG, self).__init__()
        layers = {
        'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
            }
        self.features = self.make_layers(layers[model_name])
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        feature = torch.flatten(x, 1)
        y = self.classifier(feature)
        return feature, y

    def make_layers(self, cfg, batch_norm=False):
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

def get_VGG(model_name, num_classes, dataset, pretrained = True):
    if pretrained:
        model = VGG(model_name, 1000)
        state_dict = load_state_dict_from_url(model_urls[model_name], progress=True)
        model.load_state_dict(state_dict)
        if dataset != "Imagenet":
            input = model.classifier[6].in_features
            model.classifier[6] = torch.nn.Linear(input, num_classes)
    else:
        model = VGG(model_name, num_classes)
    return model 

class MNIST_Model(nn.Module):
    def __init__(self, num_classes):
        super(MNIST_Model, self).__init__()
        in_channel = 1
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 3),
            nn.ReLU(inplace = True),
            nn.Conv2d(32, 32, kernel_size = 3),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(32, 64, kernel_size = 3),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, kernel_size = 3),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 18 * 18, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(200, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, num_classes),
        )
    
    def forward(self, img):
        feature = self.features(img)
        feature = torch.flatten(feature, 1)
        logits = self.classifier(feature)
        return feature, logits
    
# if __name__ == "__main__":
#     model = get_VGG("VGG11", 10, True)
#     # for name, module in model.named_modules():
#     #     print([name, module])
#     print(model._modules)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class Bottleneck(nn.Module):
    
    # architecture of a single bottleneck: [[1x1, 3x3, 1x1]], with each expansion = 4
    def __init__(self, in_c, out_c, downsample, stride = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size = 1, stride = 1, padding = 0)
        self.batch_norm1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size = 3, stride = stride, padding = 1)
        self.batch_norm2 = nn.BatchNorm2d(out_c)
        self.conv3 = nn.Conv2d(out_c, out_c*4, kernel_size = 1, stride = 1, padding = 0)
        self.batch_norm3 = nn.BatchNorm2d(out_c*4)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace = True)   

    
    def forward(self,x):
        
        identity = x.clone()
        z1 = self.relu(self.batch_norm1(self.conv1(x)))
        z2 = self.relu(self.batch_norm2(self.conv2(z1)))
        z3 = self.batch_norm3(self.conv3(z2))
        if self.downsample is not None:
            identity = self.downsample(identity)
          
        z3 += identity
        z = self.relu(z3)

        return z
    
class ResNet50(nn.Module):
    def __init__(self, layer_list, in_channels = 3, out_class = 100):
        super().__init__()
        # initial block input channels: 64, 128, 256, 512 (with expansion = 4)
        self.in_c = 64
        # in layer0, first two layers: 7x7, 64, stride = 2 and 3x3 max pool, stride = 2
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )

        # following layers = setting of ResNet 50 with each layer num: 3, 4, 6, 3
        self.layer1 = self.make_layer(out_c = 64, layer_num = layer_list[0], stride = 1)
        self.layer2 = self.make_layer(out_c = 128, layer_num = layer_list[1], stride = 2)
        self.layer3 = self.make_layer(out_c = 256, layer_num = layer_list[2], stride = 2)
        self.layer4 = self.make_layer(out_c = 512, layer_num = layer_list[3], stride = 2)
        
        # for calssification task
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(2048, out_class)
            #nn.Linear(2048, 512),
            #nn.ReLU(inplace = True),
            #nn.Dropout(0.4), 
            #nn.Linear(512, out_class)
        )

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0) 


    def make_layer(self, out_c, layer_num, stride):
        # every first layer need downsampling to match the size b/w x and identity(residual) size
        downsample = nn.Sequential(
            nn.Conv2d(self.in_c, out_c*4, kernel_size = 1, stride = stride),
            nn.BatchNorm2d(out_c*4)
        )
        layers = []
        layers.append(Bottleneck(self.in_c, out_c, downsample, stride))
        self.in_c = out_c*4
        for i in range(layer_num-1):
            layers.append(Bottleneck(self.in_c, out_c, None))
        return nn.Sequential(*layers)

    def forward(self, x):
        z1 = self.layer0(x)

        z2 = self.layer1(z1)
        z3 = self.layer2(z2)
        z4 = self.layer3(z3)
        z5 = self.layer4(z4)

        z = self.classifier(z5)
        return z


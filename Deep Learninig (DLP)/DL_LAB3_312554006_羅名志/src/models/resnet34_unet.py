import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Bottleneck(nn.Module):
    
    # architecture of a single bottleneck: [[1x1, 3x3, 1x1]], with each expansion = 4
    def __init__(self, in_c, out_c, downsample, stride = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size = 1, stride = 1, padding = 0)
        self.batch_norm1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size = 3, stride = stride, padding = 1)
        self.batch_norm2 = nn.BatchNorm2d(out_c)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace = True)   

    
    def forward(self,x):
        
        identity = x.clone()
        z1 = self.relu(self.batch_norm1(self.conv1(x)))
        z2 = self.relu(self.batch_norm2(self.conv2(z1)))
        
        if self.downsample is not None:
            identity = self.downsample(identity)
          
        z3 = z2 + identity
        z = self.relu(z3)

        return z
    
class ResNet34_Unet(nn.Module):
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
        
        self.layer5 =  DoubleConv(1024, 512)
        self.layer6 =  nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.layer7 =  DoubleConv(512, 256)
        self.layer8 =  nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.layer9 =  DoubleConv(256, 128)
        self.layer10 =  nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.layer11 =  DoubleConv(128, 64)
        
        self.bridge = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size = 2, stride = 2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(32, 16, kernel_size = 2, stride = 2)
        )

        self.output = nn.Sequential(
                        nn.Conv2d(16, 1, kernel_size=1),
                        nn.Sigmoid()
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
            nn.Conv2d(self.in_c, out_c, kernel_size = 1, stride = stride),
            nn.BatchNorm2d(out_c)
        )
        layers = []
        layers.append(Bottleneck(self.in_c, out_c, downsample, stride))
        self.in_c = out_c
        for i in range(layer_num-1):
            layers.append(Bottleneck(self.in_c, out_c, None))
        return nn.Sequential(*layers)


    def forward(self, x):
        x = x.float()
        z1 = self.layer0(x)
        # print(f'z1{z1.shape}')
        z2 = self.layer1(z1)
        # print(f'z2{z2.shape}')
        z3 = self.layer2(z2)
        # print(f'z3{z3.shape}')
        z4 = self.layer3(z3)
        # print(f'z4{z4.shape}')

        z5 = self.layer4(z4)
        # print(z5.shape)

        z6 = self.layer5(torch.cat([z5, z5], dim=1))
        # print(f'z6{z6.shape}')

        z7 = self.layer6(z6)
        # print(f'z7{z7.shape}')
        z8 = self.layer7(torch.cat([z7, z4], dim=1))
        # print(f'z8{z8.shape}')
        z9 = self.layer8(z8)
        # print(f'z9{z9.shape}')
        z10 = self.layer9(torch.cat([z9, z3], dim=1))
        # print(f'z10{z10.shape}')
        z11 = self.layer10(z10)
        # print(f'z11{z11.shape}')
        z12 = self.layer11(torch.cat([z11, z2], dim=1))
        # print(f'z12{z12.shape}')
        z13 = self.bridge(z12)
        # print(f'z13{z13.shape}')
        z = self.output(z13)
        # print(z.shape)
        # print('c')
        return z


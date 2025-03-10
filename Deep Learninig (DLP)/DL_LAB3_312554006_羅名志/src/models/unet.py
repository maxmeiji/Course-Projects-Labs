import torch
import torch.nn as nn

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


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            DoubleConv(in_channels, 64),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2),
                DoubleConv(64, 128)
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2),
                DoubleConv(128, 256)
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2),
                DoubleConv(256, 512)
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2),
                DoubleConv(512, 1024)
            ),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            DoubleConv(1024, 512),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            DoubleConv(512, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            DoubleConv(256, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            DoubleConv(128, 64),
            nn.Sequential(
                nn.Conv2d(64, out_channels, kernel_size=1),
                nn.Sigmoid()

            )
        )
        self._init_weight()

    def _init_weight(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = x.float()
        x1 = self.encoder[0](x)
        # print(x1.shape)
        x2 = self.encoder[1](x1)
        # print(x2.shape)
        x3 = self.encoder[2](x2)
        # print(x3.shape)
        x4 = self.encoder[3](x3)
        # print(x4.shape)
        x5 = self.encoder[4](x4)
        # print(x5.shape)
        

        x = self.decoder[0](x5)
        x = torch.cat([x, x4], dim=1)
        x = self.decoder[1](x)
        # print(x.shape)
        x = self.decoder[2](x)
        x = torch.cat([x, x3], dim=1)
        x = self.decoder[3](x)
        # print(x.shape)

        x = self.decoder[4](x)
        x = torch.cat([x, x2], dim=1)
        x = self.decoder[5](x)
        # print(x.shape)

        x = self.decoder[6](x)
        x = torch.cat([x, x1], dim=1)
        x = self.decoder[7](x)
        # print(x.shape)
       
        # output layer 
        x = self.decoder[8](x)
        # print(x.shape)

        return x
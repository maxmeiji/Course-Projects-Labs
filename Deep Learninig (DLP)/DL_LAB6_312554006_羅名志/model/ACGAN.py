import torch
import torch.nn as nn
import numpy as np

""" the overall architecture is referenced from https://github.com/clvrai/ACGAN-PyTorch/blob/master/network.py"""

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        # define args
        self.num_classes = args.num_classes
        self.latent_dim = args.latent_dim
        self.image_size = args.image_size
        self.channels = 3  # Assuming RGB images

        # define forward function
        self.label_embedding = nn.Sequential(
            nn.Linear(self.num_classes, self.num_classes),
            nn.LeakyReLU(0.2, True)
        )
        self.init_size = 24
        self.l1 = nn.Sequential(nn.Linear(self.latent_dim + self.num_classes, 128))

        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
        )
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.tconv6 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_embedding(labels), noise), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, 1, 1)
        out = self.tconv2(out)
        out = self.tconv3(out)
        out = self.tconv4(out)
        out = self.tconv5(out)
        img = self.tconv6(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        # define args
        self.num_classes = args.num_classes
        self.image_size = args.image_size
        self.channels = 3  # Assuming RGB images

        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 6
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )

        self.adv_layer = nn.Sequential(nn.Linear(512 * 5 ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(512 * 5 ** 2, self.num_classes), nn.Softmax(dim=1))

    def forward(self, img, labels):
        conv1 = self.conv1(img)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        flat6 = conv6.view(-1, 5*5*512)
        val = self.adv_layer(flat6)
        label = self.aux_layer(flat6)
        return val, label
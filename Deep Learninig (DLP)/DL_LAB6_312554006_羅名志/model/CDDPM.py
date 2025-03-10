import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import torch
import torch.nn as nn

class ConditionalUNet(nn.Module):
    def __init__(self, num_classes=24, class_emb_size=24):
        super().__init__()

        # The embedding layer will map the class label to a vector of size class_emb_size
        self.class_emb1 = nn.Sequential(
            nn.Linear(num_classes, num_classes * 2),
            nn.GELU(),
            nn.Linear(num_classes * 2, num_classes * 2)
        )
        self.class_emb2 = nn.Sequential(
            nn.Linear(num_classes * 2, num_classes),
            nn.GELU(),
            nn.Linear(num_classes, num_classes)
        )
        self.time_emb1 = nn.Sequential(
            nn.Linear(1, num_classes * 2),
            nn.GELU(),
            nn.Linear(num_classes * 2, num_classes * 2)
        )
        self.time_emb2 = nn.Sequential(
            nn.Linear(num_classes * 2, num_classes),
            nn.GELU(),
            nn.Linear(num_classes, num_classes)
        )

        # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
        self.model = UNet2DModel(
            sample_size=64,  # the target image resolution
            in_channels=3 + 1* class_emb_size,  # Additional input channels for class cond.
            out_channels=3,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(32, 64, 64),
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",  # a regular ResNet upsampling block
            ),
        )

    # Our forward method now takes the class labels as an additional argument
    def forward(self, x, t, class_labels):
        # Shape of x:
        bs, ch, w, h = x.shape

        # Convert t to float if it's not already
        t = t.float()

        # Class conditioning in right shape to add as additional input channels
        class_cond = self.class_emb1(class_labels)  # Map to embedding dimension
        class_cond = self.class_emb2(class_cond)  # Map to embedding dimension
        # time = self.time_emb1(t.unsqueeze(1))  # Expand t to (bs, 1)
        # time = self.time_emb2(time)  # Use 'time' not 't'

        class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)
        #t ime = time.view(bs, time.shape[1], 1, 1).expand(bs, time.shape[1], w, h)

        # Net input is now x and class cond concatenated together along dimension 1
        net_input = torch.cat((x, class_cond), 1)

        # Feed this to the UNet and return the prediction
        return self.model(net_input, t).sample  # Pass 't' as timestep

# Assuming UNet2DModel is defined somewhere or imported correctly.


class DDPM(nn.Module):
    def __init__(self, unet, timesteps=1000):
        super(DDPM, self).__init__()
        self.unet = unet
        self.timesteps = timesteps
        self.betas = torch.linspace(0.0001, 0.02, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)
    
    def cosine_beta_schedule(timesteps, s=0.008, device='cpu'):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, device=device)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * (np.pi / 2)) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        
        self.betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        self.alphas = 1 - betas
        self.alpha_hat = torch.cumprod(alphas, dim=0)
        
    return betas, alphas, alpha_hat
    
    def forward_diffusion(self, x0, t, noise=None):
        bc, c, h, w = x0.shape
        if noise == None:
            noise = torch.randn_like(x0)
        alpha_hat_t = self.alpha_hat[t]
        noisy = alpha_hat_t.sqrt().reshape(bc, 1, 1, 1) * x0 + (1 - alpha_hat_t).sqrt().reshape(bc, 1, 1, 1) * noise
        return noisy

    def reverse_diffusion(self, xt, t, condition):
        bc, c, h, w = xt.shape
        noise_pred = self.unet(xt, t, condition)
        alpha_hat_t = self.alpha_hat[t]
        alpha_t = self.alphas[t]
        beta_t = self.betas[t]

        
        # return noise_pred, (xt - torch.sqrt(1.0 - alpha_hat_t) * noise_pred) / torch.sqrt(alpha_hat_t)
        denomiantor = 1/(alpha_t.sqrt().reshape(bc, 1, 1, 1))
        noise_weight = (1-alpha_t.reshape(bc, 1, 1, 1))/(torch.sqrt(1-alpha_hat_t.reshape(bc, 1, 1, 1)))
        random_weight = torch.sqrt(beta_t.reshape(bc, 1, 1, 1))
        z = torch.randn_like(xt).to(device)
        return noise_pred, (denomiantor * (xt-noise_pred*noise_weight) + (z*random_weight))

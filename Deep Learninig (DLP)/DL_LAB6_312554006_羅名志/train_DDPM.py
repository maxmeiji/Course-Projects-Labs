""" 
1. The architecture of Unet is referneced from https://github.com/huggingface/diffusion-models-class/blob/main/unit2/02_class_conditioned_diffusion_model_example.ipynb
2. The architecture of Diffusion part is referenced from 
    a. https://arxiv.org/pdf/2006.11239
    b. https://medium.com/@brianpulfer/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1    
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from tqdm import tqdm
from dataloader import Dataset_iclevr
from model.CDDPM import ConditionalUNet, DDPM
import warnings 
from evaluator import evaluation_model

warnings.filterwarnings("ignore")

# TensorBoard writer
writer = SummaryWriter()
normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=24, help="number of classes for dataset")
    parser.add_argument("--image_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
    parser.add_argument("--num_epochs", type=int, default=500, help="number of epochs of training")
    parser.add_argument("--log_interval", type=int, default=1, help="interval between logging metrics to TensorBoard")
    parser.add_argument("--save_interval", type=int, default=10, help="interval between saving models")
    parser.add_argument("--sample_interval", type=int, default=1, help="interval between saving generated samples")
    parser.add_argument("--dataset_path", type=str, default="iclevr", help="path to the dataset")
    args = parser.parse_args()
    return args


# Training function
def train(args):
    # Ensure directories exist
    os.makedirs('ddpm/output_images', exist_ok=True)
    os.makedirs('ddpm/model_weight', exist_ok=True)

    # Dataloader
    train_dataloader = DataLoader(
        Dataset_iclevr(root=args.dataset_path, mode='train'),
        batch_size=args.batch_size,
        shuffle=True
    )
    test_dataloader = DataLoader(
        Dataset_iclevr(root=args.dataset_path, mode='test'),
        batch_size=args.batch_size,
        shuffle=True
    )
    
    # Initialize models
    unet = ConditionalUNet(args.num_classes, args.num_classes).to(args.device)
    ddpm = DDPM(unet).to(args.device)
    
    optimizer = optim.Adam(ddpm.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    criterion = nn.MSELoss().to(args.device)

    timesteps = 1000
    best_acc = 0.0

    for epoch in range(args.num_epochs):
        tot_loss = 0
        ddpm.train()
        for i, (imgs, labels) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}/{args.num_epochs}")):            
            batch_size = imgs.size(0)
            real_imgs = imgs.to(args.device)
            labels = labels.to(args.device)

            t = torch.randint(0, timesteps, (batch_size,), device=args.device).long()
            noise = torch.randn_like(real_imgs).to(args.device)
            xt = ddpm.forward_diffusion(real_imgs, t, noise)
            noise_pred = ddpm.unet(xt, t, labels)
            loss = criterion(noise, noise_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            tot_loss += loss.item()
        
        # Log
        writer.add_scalar('Loss/DDPM', tot_loss / len(train_dataloader), epoch)


        # Save noise addition process for the first image in the training set
        if epoch == 1:
            save_noise_addition_process(ddpm, real_imgs[0], f"ddpm/output_images/epoch_{epoch}_noise_addition_process.png")

        # Generate samples based on test samples
        ddpm.eval()
        evaluator = evaluation_model()
        test_acc = 0
        first_image_saved = False

        with torch.no_grad():
            for i, (imgs, labels) in enumerate(tqdm(test_dataloader, desc=f"Epoch {epoch}/{args.num_epochs}")):
                batch_size = labels.size(0)
                samples = sample_images(ddpm, labels, num_samples=batch_size)                    
                accuracy = evaluator.eval(samples, labels)
                # print(accuracy)
                test_acc = test_acc + accuracy

                if not first_image_saved and batch_size > 0:
                    # Save denoising process for the first image in the test set
                    save_denoising_process(ddpm, labels[3], f"ddpm/output_images/epoch_{epoch}_denoising_process.png")
                    first_image_saved = True
            if test_acc/len(test_dataloader) > best_acc:
                best_acc = test_acc/len(test_dataloader)
                print(f"Reach the best accuracy: {best_acc}")
                torch.save(ddpm.state_dict(), f"model_weight/ddpm_best.pth")

        # Log
        writer.add_scalar('Loss/DDPM', tot_loss / len(train_dataloader), epoch)
        writer.add_scalar('Test/Accuracy', test_acc/len(test_dataloader), epoch)
        
        # if epoch % args.sample_interval == 0:
        #     save_image(samples.data[:32], f"output_images/{epoch}_DDPM.png", nrow=8, normalize=True)


        print(f"[Epoch {epoch}/{args.num_epochs}] [Loss: {tot_loss / len(train_dataloader)}] [ACC: {test_acc/len(test_dataloader)}]")

def save_denoising_process(ddpm, condition, img_path, num_samples=8):
    condition = condition.view(1, 24)    
    condition = condition.to(args.device)
    xt = torch.rand((1, 3, 64, 64), device=args.device)
    step_size = ddpm.timesteps // num_samples
    denoising_steps = [xt.clone().detach().cpu()]

    for t in reversed(range(ddpm.timesteps)):
        with torch.no_grad():
            n, xt = ddpm.reverse_diffusion(xt, torch.tensor(t, device=args.device), condition)
            if t % step_size == 0 or t == 0:
                denoising_steps.append(xt.clone().detach().cpu())


    denoising_steps = torch.cat(denoising_steps, dim=0)
    save_image(denoising_steps, img_path, nrow=num_samples+1, normalize=True)


def sample_images(ddpm, condition, num_samples=1):
    ddpm.eval()
    condition = condition.to(args.device)
    xt = torch.randn((num_samples, 3, 64, 64), device=args.device)

    for t in reversed(range(ddpm.timesteps)):
        t_tensor = torch.full((num_samples,), t, dtype=torch.long, device=args.device)
        n, xt = ddpm.reverse_diffusion(xt, t_tensor, condition)

    return xt

def save_noise_addition_process(ddpm, real_image, img_path, num_samples=8):
    ddpm.eval()
    real_image = real_image.to(args.device).unsqueeze(0)
    noise = torch.randn_like(real_image).to(args.device)
    
    step_size = ddpm.timesteps // num_samples
    noisy_steps = []
    
    for t in range(0, ddpm.timesteps, step_size):
        xt = ddpm.forward_diffusion(real_image, torch.tensor([t], device=args.device), noise)
        noisy_steps.append(xt.clone().detach().cpu())
    
    noisy_steps = torch.cat(noisy_steps, dim=0)
    save_image(noisy_steps, img_path, nrow=num_samples+1, normalize=True)

if __name__ == '__main__':
    args = parse_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')
    args.device = device
    train(args)
    writer.close()

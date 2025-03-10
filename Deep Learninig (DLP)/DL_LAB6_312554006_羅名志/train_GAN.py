import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from model.ACGAN import Generator, Discriminator
from dataloader import Dataset_iclevr
from tqdm import tqdm
from evaluator import evaluation_model

# TensorBoard writer
writer = SummaryWriter()
normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
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
    parser.add_argument("--sample_interval", type=int, default=50, help="interval between saving generated samples")
    parser.add_argument("--dataset_path", type=str, default="path_to_dataset", help="path to the dataset")
    args = parser.parse_args()
    return args

# Training function
def train(args):
    # Ensure directories exist
    os.makedirs('output_images', exist_ok=True)
    os.makedirs('model_weight', exist_ok=True)

    # Dataloader
    train_dataloader = DataLoader(
        Dataset_iclevr(root='./iclevr', mode='train'),
        batch_size=args.batch_size,
        shuffle=True
    )
    test_dataloader = DataLoader(
        Dataset_iclevr(root='./iclevr', mode='test'),
        batch_size=args.batch_size,
        shuffle=True
    )
    
    # Initialize generator and discriminator
    generator = Generator(args).to(args.device)
    discriminator = Discriminator(args).to(args.device)
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    adversarial_loss = nn.BCELoss().to(args.device)
    auxiliary_loss = nn.BCELoss().to(args.device)

    best_acc = 0.0
    for epoch in range(args.num_epochs):
        tot_loss_d = 0
        tot_loss_g = 0
        tot_acc = 0
        generator.train()
        discriminator.train()
        evaluator = evaluation_model()
        for i, (imgs, labels) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}/{args.num_epochs}")):            
            batch_size = imgs.size(0)
            real_imgs = imgs.to(args.device)
            labels = labels.to(args.device)

            # Train Discriminator
            optimizer_D.zero_grad()
            valid = 0.4 * torch.rand(batch_size, 1, device=args.device, requires_grad=False) + 0.6
            real_pred, real_aux = discriminator(real_imgs, labels)
            d_real_loss = (adversarial_loss(real_pred, valid) + 100*auxiliary_loss(real_aux, labels))/101
            d_real_loss.backward()

            # Fake images
            fake = 0.4 * torch.rand(batch_size, 1, device=args.device, requires_grad=False)
            fake_noise = torch.randn((batch_size, args.latent_dim), device=args.device)

            fake_img = generator(fake_noise, labels)
            fake_pred, fake_aux = discriminator(fake_img.detach(), labels)
            d_fake_loss = (adversarial_loss(fake_pred, fake) + 100*auxiliary_loss(fake_aux, labels)) / 101
            d_fake_loss.backward()
            # Total loss
            d_loss = 0.5 * (d_real_loss + d_fake_loss)
            # d_loss.backward()
            optimizer_D.step()

            accuracy = evaluator.compute_acc(real_aux, labels)

            # Train Generator
            optimizer_G.zero_grad()
            fake_noise = torch.randn((batch_size, args.latent_dim), device=args.device)
            fake_img = generator(fake_noise, labels)

            # Loss measures generator's ability to fool the discriminator
            validity, pred_label = discriminator(fake_img, labels)
            valid = torch.ones(batch_size, 1, device=args.device, requires_grad=False)
            g_loss = (adversarial_loss(validity, valid) + 100*auxiliary_loss(pred_label, labels)) / 101

            g_loss.backward()
            optimizer_G.step()
            
            tot_loss_d = tot_loss_d + d_loss.item()
            tot_loss_g = tot_loss_g + g_loss.item()
            tot_acc = tot_acc + accuracy
        
        # Evaluate
        generator.eval()
        discriminator.eval()
        evaluator = evaluation_model()
        test_acc = 0
        with torch.no_grad():
            for i, (imgs, labels) in enumerate(tqdm(test_dataloader, desc=f"Epoch {epoch}/{args.num_epochs}")):
                batch_size = imgs.size(0)
                labels = labels.to(args.device)
                noise = torch.randn((batch_size, args.latent_dim), device=args.device)
                fake_img = generator(noise, labels)
                # fake_img_normalized = normalize(fake_img)
                accuracy = evaluator.eval(fake_img, labels)
                # print(accuracy)
                test_acc = test_acc + accuracy

            if test_acc/len(test_dataloader) > best_acc:
                best_acc = test_acc/len(test_dataloader)
                print(f"Reach the best accuracy: {best_acc}")
                torch.save(generator.state_dict(), f"model_weight/generator_best.pth")
                torch.save(discriminator.state_dict(), f"model_weight/discriminator_best.pth")
        # Log
        writer.add_scalar('Loss/Generator', tot_loss_g/len(train_dataloader), epoch)
        writer.add_scalar('Loss/Discriminator', tot_loss_d/len(train_dataloader), epoch)
        writer.add_scalar('Train/Accuracy', tot_acc/len(train_dataloader), epoch)
        writer.add_scalar('Test/Accuracy', test_acc/len(test_dataloader), epoch)

    
        if epoch % args.sample_interval == 0:
            save_image(fake_img.data[:25], f"output_images/{epoch}.png", nrow=5, normalize=True)

        print(f"[Epoch {epoch}/{args.num_epochs}] [D loss: {tot_loss_d/len(train_dataloader)}] [G loss: {tot_loss_g/len(train_dataloader)}] [Train acc: {tot_acc/len(train_dataloader)}] [Test acc: {test_acc/len(test_dataloader)}]")

if __name__ == '__main__':
    args = parse_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')
    args.device = device
    train(args)
    writer.close()

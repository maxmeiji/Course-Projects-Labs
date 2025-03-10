import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from model.ACGAN import Generator, Discriminator
from dataloader import Dataset_iclevr
from tqdm import tqdm
from evaluator import evaluation_model
from torchvision.utils import save_image, make_grid
import warnings
warnings.filterwarnings("ignore")



def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--num_classes", type=int, default=24, help="number of classes for dataset")
    parser.add_argument("--image_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
    parser.add_argument("--num_epochs", type=int, default=300, help="number of epochs of training")
    parser.add_argument("--log_interval", type=int, default=1, help="interval between logging metrics to TensorBoard")
    parser.add_argument("--save_interval", type=int, default=10, help="interval between saving models")
    parser.add_argument("--sample_interval", type=int, default=50, help="interval between saving generated samples")
    parser.add_argument("--dataset_path", type=str, default="./model_weight", help="path to the dataset")
    args = parser.parse_args()
    return args


def test(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'    
    generator = Generator(args).to(args.device)
    discriminator = Discriminator(args).to(args.device)
    generator.load_state_dict(torch.load(os.path.join(args.dataset_path, 'generator_best.pth')))
    discriminator.load_state_dict(torch.load(os.path.join(args.dataset_path, 'discriminator_best.pth')))

    test_dataloader = DataLoader(
        Dataset_iclevr(mode='test', root='iclevr'),
        batch_size=args.batch_size,
        shuffle=False
    )

    new_test_dataloader = DataLoader(
        Dataset_iclevr(mode='new_test', root='iclevr'),
        batch_size=args.batch_size,
        shuffle=False
    )

    # Evaluate
    generator.eval()
    discriminator.eval()
    evaluator = evaluation_model()
    test_acc = 0
    new_test_acc = 0

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(tqdm(test_dataloader)):
            batch_size = imgs.size(0)
            labels = labels.to(args.device)
            noise = torch.randn((batch_size, args.latent_dim), device=args.device)
            fake_img = generator(noise, labels)
            accuracy = evaluator.eval(fake_img, labels)

            test_acc = test_acc + accuracy 
            grid = make_grid(fake_img[:32], nrow=8, normalize=True)
            save_image(grid, f"output_images/test.png")

        for i, (imgs, labels) in enumerate(tqdm(new_test_dataloader)):
            batch_size = imgs.size(0)
            labels = labels.to(args.device)
            noise = torch.randn((batch_size, args.latent_dim), device=args.device)
            fake_img = generator(noise, labels)
            accuracy = evaluator.eval(fake_img, labels)

            new_test_acc = new_test_acc + accuracy    
            grid = make_grid(fake_img[:32], nrow=8, normalize=True)
            save_image(grid, f"output_images/new_test.png")

    print(f'Accuracy for test.json: {test_acc}')
    print(f'Accuracy for new_test.json: {new_test_acc}')
    #print(test_acc, new_test_acc)

if __name__ == "__main__":
    args = parse_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device
    test(args)
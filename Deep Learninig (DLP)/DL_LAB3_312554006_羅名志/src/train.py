import argparse
from models.unet import UNet
from oxford_pet import load_dataset
import os 
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import dice_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
import torch.nn as nn
from evaluate import evaluate
from models.resnet34_unet import ResNet34_Unet

def train(args):
    # implement the training function here
    print("...........start training...........")
    print(f"model: {args.model}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')
    
    # load data
    if args.data_path == 'no path':
        curr_path = os.getcwd()
        data_path = os.path.join(curr_path, os.path.join('dataset','os.oxford-iiit-pet/'))    
    else:
        data_path = args.data_path
    train_dataset = load_dataset(data_path, 'train')
    print(f"Successfully load training dataset: with {len(train_dataset)} data")
    valid_dataset = load_dataset(data_path, 'valid')
    print(f"Successfully load validation dataset: with {len(valid_dataset)} data")

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size = args.batch_size, shuffle=True)

    # initialize recording variables
    writer = SummaryWriter(filename_suffix = 'unet')
    best_valid_loss = float('inf')
    best_model_weights = None

    # define the model
    if args.model == 'Unet':
        model = UNet(in_channels=3, out_channels=1)
    else:
        model = ResNet34_Unet([3,4,6,3], 3, 1)
    torch.autograd.set_detect_anomaly(True) 
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCELoss()
    curr_path = os.getcwd()
    pth = os.path.join(curr_path, os.path.join('saved_models',f'{args.model}.pth'))
    # train part
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_dice = 0.0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave = False) as t:
            for sample in t:
                img, mask = sample["image"].to(device), sample["mask"].to(device)
                # print(img.shape, img.dtype)
                # print(mask.shape, mask.dtype)
                # zero the parameters gradients
                optimizer.zero_grad()

                # foward
                outputs = model(img)
                outputs = outputs.squeeze(1)
                mask = mask.squeeze(1)
                # print(outputs[0], mask[0])
                loss = criterion(outputs, mask)
                dice = dice_score(outputs, mask)
                # print(dice)
                # backward
                loss.backward()
                optimizer.step()

                # update the progress bar with current loss
                running_loss += loss.item()*img.size(0)
                running_dice += dice.item()*img.size(0)
                t.set_postfix(loss=loss.item())
        
    
        
        # recore training acc and loss
        train_loss, train_dice = running_loss/len(train_loader.dataset), running_dice/len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {train_loss:.4f}, Training dice score: {train_dice:.4f}")
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Dice_score/train', train_dice, epoch)

        
        # Evaluate on the validation set
        valid_loss, valid_dice = evaluate(model, valid_loader, device)
        print(f"Epoch [{epoch+1}/{args.epochs}], Validation Loss: {valid_loss:.4f}, Validation dice score: {valid_dice:.4f}")
        writer.add_scalar('Loss/valid', valid_loss, epoch)
        writer.add_scalar('Dice_score/valid', valid_dice, epoch)
 
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_weights = model.state_dict()
          
    torch.save(best_model_weights, pth)
    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, default= 'no path', help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--model', type=str, default='ResNet34_Unet', help='model selection')
    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args)
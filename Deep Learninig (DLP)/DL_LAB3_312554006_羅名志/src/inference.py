import argparse
import os 
from models.unet import UNet
from models.resnet34_unet import ResNet34_Unet
import torch
from oxford_pet import load_dataset
from evaluate import evaluate
from torch.utils.data import DataLoader
from evaluate import instance_visualzie
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='saved_models/ResNet34_Unet.pth', help='path to the stored model weoght')
    # parser.add_argument('--data_path', type=str, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='batch size')
    parser.add_argument('--model_name', '-m', type=str, default='ResNet34_Unet', help='batch size')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')
    
    # get model path
    curr_path = os.getcwd()
    data_path = os.path.join(curr_path, os.path.join('dataset','os.oxford-iiit-pet/'))    

    weight_path = os.path.join(curr_path, args.model)
    
    # load dataset
    train_dataset = load_dataset(data_path, 'train')
    print(f"Successfully load training dataset: with {len(train_dataset)} data")
    valid_dataset = load_dataset(data_path, 'valid')
    print(f"Successfully load validation dataset: with {len(valid_dataset)} data")
    test_dataset = load_dataset(data_path, 'test')
    print(f"Successfully load testing dataset: with {len(test_dataset)} data")
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size = args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False)


    # Define the model 
    if args.model_name == 'Unet':
        model = UNet(in_channels=3, out_channels=1)
    else:
        model = ResNet34_Unet([3,4,6,3], 3, 1)

    
    model.load_state_dict(torch.load(weight_path))
    train_loss, train_dice = evaluate(model, train_loader, device)
    valid_loss, valid_dice = evaluate(model, valid_loader, device)
    test_loss, test_dice = evaluate(model, test_loader, device)

    
    # Print the result
    print(f"Model Name: {args.model_name}, Model Path: {weight_path}")
    print(f"Train Loss: {train_loss:.4f}, Test Dice Score: {train_dice:.4f}")
    print(f"Valid Loss: {valid_loss:.4f}, Valid Dice Score: {valid_dice:.4f}")
    print(f"Test Loss: {test_loss:.4f}, Test Dice Score: {test_dice:.4f}")

    
    # visualize the segmentation result
    instance_input, instance_output, instance_truth = instance_visualzie(model, test_loader, device)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot
    instance_input = instance_input.cpu().numpy()
    instance_input = np.transpose(instance_input, (1, 2, 0))
    axes[0].imshow(instance_input)  # Assuming instance input is grayscale
    axes[0].set_title('Instance Input')
    axes[0].axis('off')

    instance_output = instance_output.cpu().numpy()
    instance_output = np.transpose(instance_output, (1, 2, 0))
    axes[1].imshow(instance_output, cmap='gray')  # Assuming instance output is grayscale
    axes[1].set_title('Instance Output')
    axes[1].axis('off')

    instance_truth = instance_truth.cpu().numpy()
    instance_truth = np.transpose(instance_truth, (1, 2, 0))
    axes[2].imshow(instance_truth, cmap='gray')  # Assuming instance truth is grayscale
    axes[2].set_title('Instance Truth')
    axes[2].axis('off')

    plt.savefig(f'{args.model_name}-result.png')
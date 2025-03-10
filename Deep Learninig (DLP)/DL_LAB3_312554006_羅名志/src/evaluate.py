import torch
from utils import dice_score
import torch.nn as nn

def instance_visualzie(net, data, device):
    net.to(device)
    net.eval()
    with torch.no_grad():
        for sample in data:
            inputs, masks = sample["image"].to(device), sample["mask"].to(device)
        
            outputs = net(inputs)
            outputs[0] = (outputs[0] > 0.5)
                
            return inputs[0], outputs[0], masks[0]

def evaluate(net, data, device):
    # implement the evaluation function here
    net.to(device)
    net.eval()
    criterion = nn.BCELoss()
    total_loss = 0.0
    total_dice = 0.0
    
    with torch.no_grad():
        for sample in data:
            inputs, masks = sample["image"].to(device), sample["mask"].to(device)
            outputs = net(inputs)
            loss = criterion(outputs, masks)
            dice = dice_score(outputs, masks)

            total_loss += loss.item() * inputs.size(0)
            total_dice += dice.item() * inputs.size(0)
            
    avg_loss, avg_dice = total_loss/len(data.dataset), total_dice/len(data.dataset)
    
    return avg_loss, avg_dice
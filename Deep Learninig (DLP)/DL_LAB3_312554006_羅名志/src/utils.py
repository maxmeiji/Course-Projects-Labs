import torch

def dice_score(pred_mask, gt_mask):
    # implement the Dice score here
    #Dice score = 2 * (number of common pixels) / (predicted img size + groud truth img size)
    pred_mask = (pred_mask > 0.5)
    gt_mask = gt_mask

    intersection = torch.sum(pred_mask*gt_mask)
    dice = (2.0 * intersection) / (torch.sum(pred_mask) + torch.sum(gt_mask))

    return dice


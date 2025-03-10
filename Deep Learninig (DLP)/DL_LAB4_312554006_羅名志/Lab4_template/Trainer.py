import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10
from torch.utils.tensorboard import SummaryWriter

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        # TODO
        self.mode = args.kl_anneal_type
        self.cycle = args.kl_anneal_cycle
        self.ratio = args.kl_anneal_ratio
        self.thres_monotonic = args.num_epoch*0.25
        # per cycle epoch restart a cycle
        self.cycle_epoch = args.num_epoch/self.cycle
        self.thres_cyclical = 0.5*self.cycle_epoch
        self.weight = 1
        self.current_epoch = current_epoch
        # raise NotImplementedError
        
    def update(self):
        # TODO
        # first updata the itereation progress
        self.current_epoch = self.current_epoch + 1

        if self.mode == 'Monotonic':
            if self.current_epoch < self.thres_monotonic:
                self.weight = 1/(self.thres_monotonic)*self.current_epoch
            else:
                self.weight = 1
        
        elif self.mode == "Cyclical":
            if self.current_epoch%self.cycle_epoch < self.thres_cyclical:
                self.weight = 1/(self.thres_cyclical)* (self.current_epoch%self.cycle_epoch) 
            else:
                self.weight = 1       
        else:
            self.weight = 1
        #raise NotImplementedError
    
    def get_beta(self):
        # TODO
        return self.weight

    #def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=1):
    #    # #TODO
    #    raise NotImplementedError




class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
        
    def forward(self, img, label):
        pass
    
    def training_stage(self):
        # for visualization
        writer = SummaryWriter(log_dir="./logs")
        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
 
                loss = self.training_one_step(img, label, adapt_TeacherForcing)
                
                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                
                
            writer.add_scalar('Loss/train', loss.item(), i)
            writer.add_scalar('TeacherForcing_ratio', self.tfr, i)
            writer.add_scalar('Beta', beta, i)
        
            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))
                
            _, val_psnr = self.eval()
            writer.add_scalar('Loss/PSNR', val_psnr, i)
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
        writer.close()  
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss, psnr = self.val_one_step(img, label)
            self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
        return loss, psnr


    def training_one_step(self, img, label, adapt_TeacherForcing):
        # TODO
        img = img.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        label = label.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        # print(img[0].size()) # ([2,3,32,64]: [B, C, H, W])
        # first extract all 
        H_frame = []
        H_label = []
        for i in range(self.train_vi_len):
            H_frame.append(self.frame_transformation(img[i]))
            H_label.append(self.label_transformation(label[i]))
        H_pred = []
        H_pred.append(H_frame[0])
        # use the firsdt frame to predict the following frame in VAE architecture
        recon_loss = 0
        kl_loss = 0
        for i in range(1,self.train_vi_len):
            z, mu, logvar = self.Gaussian_Predictor(H_frame[i], H_label[i])
            if adapt_TeacherForcing==True:
                fusion = self.Decoder_Fusion(H_frame[i-1], H_label[i], z)
            else:
                fusion = self.Decoder_Fusion(H_pred[i-1], H_label[i], z)
            output = self.Generator(fusion)
            output_trans = self.frame_transformation(output)
            # print(output.shape, output_trans.shape, H_pred[0].shape)
            H_pred.append(output_trans)
            
            recon_loss += self.mse_criterion(output, img[i])
            kl_loss += kl_criterion(mu, logvar, self.batch_size)
            
        beta = self.kl_annealing.get_beta()
        
        loss = recon_loss + beta*kl_loss
        # print("---------")
        # print(beta, recon_loss, kl_loss)
        # print(loss)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss
    
        # raise NotImplementedError
    
    def val_one_step(self, img, label):
        # TODO
        img = img.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        label = label.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        # first extract all 
        H_frame = []
        H_label = []
        for i in range(self.val_vi_len):
            H_frame.append(self.frame_transformation(img[i]))
            H_label.append(self.label_transformation(label[i]))
        H_pred = []
        H_pred.append(H_frame[0])
        # use the firsdt frame to predict the following frame in VAE architecture
        recon_loss = 0
        kl_loss = 0
        psnr_values = []
        for i in range(1,self.val_vi_len):
            # encoder will be useless when validation, just want to know the size of z
            z, mu, logvar = self.Gaussian_Predictor(H_frame[i], H_label[i])
            z = torch.randn_like(z)            
            fusion = self.Decoder_Fusion(H_pred[i-1], H_label[i], z)
            output = self.Generator(fusion)
            output_trans = self.frame_transformation(output)
            # print(output.shape, output_trans.shape, H_pred[0].shape)
            H_pred.append(output_trans)
            
            recon_loss += self.mse_criterion(output, img[i])
            kl_loss += kl_criterion(mu, logvar, self.batch_size)
            
            psnr = Generate_PSNR(output, img[i])  # Assuming Generate_PSNR is defined correctly
            psnr_values.append(psnr.item())
        avg_PSNR = np.mean(psnr_values) 
        print(f'Vaidation of {self.current_epoch} average PSNR: {avg_PSNR}')  
        beta = self.kl_annealing.get_beta()
        loss = recon_loss + beta*kl_loss

        # plot PSNR
        plt.figure(figsize=(10, 5))
        plt.plot(psnr_values, label='PSNR per Frame')
        plt.xlabel('Frame')
        plt.ylabel('PSNR')
        plt.title('PSNR per Frame in Validation Dataset')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"./psnr/psnr_epoch_{self.current_epoch}.png")
        plt.close()

        return loss, avg_PSNR


                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        # TODO
        if self.current_epoch < self.tfr_sde:
            self.tfr = self.tfr
        else:
            self.tfr = max(self.tfr-self.tfr_d_step*(self.current_epoch - self.tfr_sde), 0.0)
        # raise NotImplementedError
            
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()



def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="AdamW")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=2)
    parser.add_argument('--num_epoch',     type=int, default=40,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=2,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=0.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=0,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=10,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='none',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1.5,              help="")
    

    # python Trainer.py --DR ./Lab4_Dataset --save_root ./save --fast_train

    args = parser.parse_args()
    
    main(args)

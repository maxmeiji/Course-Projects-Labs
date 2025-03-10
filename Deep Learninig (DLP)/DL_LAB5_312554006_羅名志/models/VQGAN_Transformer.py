import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    
    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        latent_map, indicies, loss = self.vqgan.encode(x)

        return latent_map, indicies.view(x.shape[0], -1)


        # raise Exception('TODO2 step1-1!')
        # return None
    
##TODO2 step1-2:    
    def gamma_func(self, mode="linear"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            return lambda ratio: ratio
        elif mode == "cosine":
            return lambda ratio: 1 - (0.5 * (1.0 + math.cos(math.pi*ratio)))

        elif mode == "square":
            return lambda ratio: ratio**2

        else:
            raise NotImplementedError

##TODO2 step1-3:            
    def forward(self, x):
        latent, indicies = self.encode_to_z(x)
        z_indices = indicies  # shape [10, 256]

        # number of each instance masking elements
        mask_num = math.floor(self.gamma(np.random.uniform()) * z_indices.shape[1])

        modified_indices = z_indices.clone()
        for i in range(z_indices.shape[0]):
            mask_indices = np.random.choice(z_indices.shape[1], mask_num, replace = False)
            modified_indices[i, mask_indices] = self.mask_token_id 

        logits = self.transformer(modified_indices)  #transformer predict the probability of tokens
        # shape of logits: b,c, 1025 (prediction) z_indicies is b, c
        return logits.permute(0,2,1), z_indices
    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, z_indices, mask, total_num, ratio):
        
        z_ind = z_indices.clone()
        z_ind[mask] = self.mask_token_id
        logits = self.transformer(z_ind)

        #Apply softmax to convert logits into a probability distribution across the last dimension.
        logits = torch.nn.functional.softmax(logits, dim=-1)

        #FIND MAX probability for each token value
        z_indices_predict_prob, z_indices_predict = torch.max(logits, dim=-1)
        
        g = torch.empty_like(z_indices_predict_prob).exponential_(1)
        temperature = self.choice_temperature * (1 - ratio)
        confidence = z_indices_predict_prob + temperature * g
        
        #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        #sort the confidence for the rank 
        confidence = torch.where(mask, confidence, torch.tensor(float('-inf')))
        _, sorted_indices = torch.sort(confidence, descending=True)
        #define how much the iteration remain predicted tokens by mask scheduling
        #At the end of the decoding process, add back the original token values that were not masked to the predicted tokens
        mask_num = int((1-self.gamma(ratio))*total_num)
        mask_bc = torch.zeros_like(mask, dtype=torch.bool)        
        mask_bc.scatter_(dim=1, index=sorted_indices[:, :mask_num], value=True)
        
        z_indices_predict = torch.where(mask_bc, z_indices_predict, z_indices)  # Define original_token_values accordingly
        # print(self.gamma(ratio),total_num,  mask_num)
        # print("----------------------")
        
        return z_indices_predict, mask_bc, self.gamma(ratio)


__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        

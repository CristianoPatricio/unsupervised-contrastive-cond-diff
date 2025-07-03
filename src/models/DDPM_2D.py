from src.models.modules.cond_DDPM import GaussianDiffusion
from src.models.modules.OpenAI_Unet import UNetModel as OpenAI_UNet
from src.models.modules.DDPM_encoder import get_encoder
import torch
from torch.nn.functional import normalize
from src.utils.utils_eval import _test_step, _test_end, get_eval_dictionary, _predict_step
import numpy as np
from pytorch_lightning.core.lightning import LightningModule
import torch.optim as optim
from typing import Any
import torchio as tio
from src.utils.generate_noise import gen_noise
import wandb
from omegaconf import open_dict
from collections import OrderedDict
from src.models.LDM.modules.diffusionmodules.util import timestep_embedding
from src.utils.utils import get_checkpoint_encoder
import os


class DDPM_2D(LightningModule):
    def __init__(self,cfg,prefix=None,fold=None):
        super().__init__()
        
        # config is related to the model (DDPM_2D.yaml)
        self.cfg = cfg
        self.fold = fold
        
        # conditioning net
        if cfg.get('condition',True):
            # with open_dict(self.cfg):
            #    self.cfg['cond_dim'] = cfg.get('unet_dim',64) * 4
            
            if cfg.get('encoder_mode') in ["default", "unfreeze"]:
                # Get encoder model
                self.encoder, out_features = get_encoder(cfg, self.fold)
            else:
                raise ValueError(f"The encoder mode {cfg.get('encoder_mode')} does not exist.")
            
        else: 
            out_features = None

        model = OpenAI_UNet(
                            image_size =  (int(cfg.imageDim[0] / cfg.rescaleFactor),int(cfg.imageDim[1] / cfg.rescaleFactor)),
                            in_channels = 1,
                            model_channels = cfg.get('unet_dim',64),
                            out_channels = 1,
                            num_res_blocks = cfg.get('num_res_blocks',3),
                            attention_resolutions = tuple(cfg.get('att_res',[3,6,12])), # 32, 16, 8
                            dropout=cfg.get('dropout_unet',0), # default is 0.1
                            channel_mult=cfg.get('dim_mults',[1, 2, 4, 8]),
                            conv_resample=True,
                            dims=2,
                            num_classes=out_features,
                            use_checkpoint=False,
                            use_fp16=True,
                            num_heads=1,
                            num_head_channels=64,
                            num_heads_upsample=-1,
                            use_scale_shift_norm=True,
                            resblock_updown=True,
                            use_new_attention_order=True,
                            use_spatial_transformer=cfg.get('spatial_transformer',False),    
                            transformer_depth=1,                            
                            )
        model.convert_to_fp16()

        timesteps = cfg.get('timesteps',1000)
        sampling_timesteps = cfg.get('sampling_timesteps',timesteps)
        self.test_timesteps = cfg.get('test_timesteps',150) 

        self.diffusion = GaussianDiffusion(
        model,
        image_size = (int(cfg.imageDim[0] / cfg.rescaleFactor),int(cfg.imageDim[1] / cfg.rescaleFactor)), # only important when sampling
        timesteps = timesteps,   # number of steps
        sampling_timesteps = sampling_timesteps,
        objective = cfg.get('objective','pred_x0'), # pred_noise or pred_x0
        channels = 1,
        loss_type = cfg.get('loss','l1'),    # L1 or L2
        p2_loss_weight_gamma = cfg.get('p2_gamma',0),
        cfg=cfg
        )
        
        if cfg.get('pretrained_encoder',False): # load pretrained encoder from cfg.modelpath
            
            assert cfg.get('encoder_path',None) is not None

            # Select the best ckpt for each fold
            ckpts = get_checkpoint_encoder(self.cfg, cfg.encoder_path)
            selected_checkpoint = ckpts[f"fold-{self.fold+1}"]
            print('Loading pretrained encoder from: ', selected_checkpoint)

            state_dict_pretrained = torch.load(selected_checkpoint)['state_dict']
            new_statedict = OrderedDict()
            for key in zip(state_dict_pretrained): 
                if 'slice_encoder' in key[0] :
                    new_key = 'slice_encoder'+ key[0].split('encoder')[-1]
                    new_statedict[new_key] = state_dict_pretrained[key[0]]
                elif 'sparse_encoder' in key[0] :
                    if not 'fc.weight' in key[0] and not 'fc.bias' in key[0]: # remove fc layer
                        new_key = 'encoder' + key[0].split('sp_cnn')[-1]
                        new_statedict[new_key] = state_dict_pretrained[key[0]]
                else:
                    new_statedict[key[0]] = state_dict_pretrained[key[0]]
            self.encoder.load_state_dict(new_statedict,strict=False)

        self.prefix = prefix
        
        self.save_hyperparameters()

    def forward(self, x): # encode input
        if self.cfg.get('condition',True):

            if self.cfg.get("encoder_mode") == "end2end":
                loss, feats = self.encoder(x)
            else:
                # NOTE: output is 128 dim
                x = self.encoder(x)
        else: 
            x = None
        
        if self.cfg.get("encoder_mode") == "end2end":
            return loss, feats
        else:
            return x


    def training_step(self, batch, batch_idx: int):

        # for debugging

        if self.cfg.get("encoder_mode") == "end2end":
            input = batch['orig'].cuda()
            loss_encoder, features = self(batch)
        else:
            # Process batch. Input shape is of (B, 1, 96, 96)
            input = batch['vol'][tio.DATA].squeeze(-1)

            # Get condition features from the encoder for Diff model guidance  
            # features shape is of (B, 128)   
            features = self(input)

        # generate noise
        if self.cfg.get('noisetype') is not None:
            # noise is of shape (32, 1, 96, 96)
            noise = gen_noise(self.cfg, input.shape).to(self.device)
        else: 
            noise = None
            
        # reconstruct
        # reco is the reconstructed image of shape (32, 1, 96, 96)
        # loss is loss.mean()
        loss_recons, reco = self.diffusion(input, cond = features, noise = noise)

        if self.cfg.get("encoder_mode") == "end2end":
            # Combine encoder loss + reconstruction loss
            total_loss = loss_encoder + loss_recons
        else:
            total_loss = loss_recons

        self.log(f'{self.prefix}train/Loss', total_loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=input.shape[0],sync_dist=True)
        return {"loss": total_loss}
    
    def validation_step(self, batch: Any, batch_idx: int):
        if self.cfg.get("encoder_mode") == "end2end":
            input = batch['orig'].cuda()
            loss_encoder, features = self(batch)
        else:
            # Process batch. Input shape is of (B, 1, 96, 96)
            input = batch['vol'][tio.DATA].squeeze(-1)

            # Get condition features from the encoder for Diff model guidance  
            # features shape is of (B, 128)   
            features = self(input)

        # generate noise
        if self.cfg.get('noisetype') is not None:
            noise = gen_noise(self.cfg, input.shape).to(self.device)
        else: 
            noise = None
        # reconstruct
        loss_recons, reco = self.diffusion(input,cond=features,noise=noise)

        if self.cfg.get("encoder_mode") == "end2end":
            # Combine encoder loss + reconstruction loss
            total_loss = loss_encoder + loss_recons
        else:
            total_loss = loss_recons

        self.log(f'{self.prefix}val/Loss_comb', total_loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=input.shape[0],sync_dist=True)
        return {"loss": total_loss}

    def on_test_start(self):
        self.eval_dict = get_eval_dictionary()
        self.inds = []
        self.latentSpace_slice = []
        self.new_size = [160,190,160]
        self.diffs_list = []
        self.seg_list = []
        if not hasattr(self,'threshold'):
            self.threshold = {}

    def test_step(self, batch: Any, batch_idx: int):
        self.dataset = batch['Dataset']
        input = batch['vol'][tio.DATA]
        data_orig = batch['vol_orig'][tio.DATA]
        data_seg = batch['seg_orig'][tio.DATA] if batch['seg_available'] else torch.zeros_like(data_orig)
        data_mask = batch['mask_orig'][tio.DATA]
        ID = batch['ID']
        age = batch['age']
        self.stage = batch['stage']
        label = batch['label']
        AnomalyScoreComb = []
        AnomalyScoreReg = []
        AnomalyScoreReco = []
        latentSpace = []
        condition_features = []

        if self.cfg.get('num_eval_slices', input.size(4)) != input.size(4):
            num_slices = self.cfg.get('num_eval_slices', input.size(4))  # number of center slices to evaluate. If not set, the whole Volume is evaluated
            start_slice = int((input.size(4) - num_slices) / 2)
            input = input[...,start_slice:start_slice+num_slices]
            data_orig = data_orig[...,start_slice:start_slice+num_slices] 
            data_seg = data_seg[...,start_slice:start_slice+num_slices]
            data_mask = data_mask[...,start_slice:start_slice+num_slices]
            ind_offset = start_slice
        else: 
            ind_offset = 0 

        final_volume = torch.zeros([input.size(2), input.size(3), input.size(4)], device = self.device)


        # reorder depth to batch dimension
        assert input.shape[0] == 1, "Batch size must be 1"
        input = input.squeeze(0).permute(3,0,1,2) # [B,C,H,W,D] -> [D,C,H,W]

        # calc features for guidance
        if self.cfg.get("encoder_mode") == "end2end":
            _, features = self(batch)
        else:
            features = self(input)

        features_single = features

        if self.cfg.condition:
            latentSpace.append(features_single.mean(0).squeeze().detach().cpu())
        else: 
            latentSpace.append(torch.tensor([0],dtype=float).repeat(input.shape[0]))

        if self.cfg.get('noise_ensemble',False): # evaluate with different noise levels
            timesteps = self.cfg.get('step_ensemble',[250,500,750]) # timesteps to evaluate
            reco_ensemble = torch.zeros_like(input)
            for t in timesteps:
                # generate noise
                if self.cfg.get('noisetype') is not None:
                    noise = gen_noise(self.cfg, input.shape).to(self.device)
                else: 
                    noise = None
                loss_diff, reco = self.diffusion(input,cond=features,t=t-1,noise=noise)
                reco_ensemble += reco
                
            reco = reco_ensemble / len(timesteps) # average over timesteps
        else :
            if self.cfg.get('noisetype') is not None:
                noise = gen_noise(self.cfg, input.shape).to(self.device)
            else: 
                noise = None
            loss_diff, reco = self.diffusion(input,cond=features,t=self.test_timesteps-1,noise=noise)
        
        # calculate loss and Anomalyscores
        AnomalyScoreComb.append(loss_diff.cpu())
        AnomalyScoreReg.append(loss_diff.cpu())
        AnomalyScoreReco.append(loss_diff.cpu())

        # reassamble the reconstruction volume
        final_volume = reco.clone().squeeze()
        final_volume = final_volume.permute(1,2,0) # to HxWxD

        # average across slices to get volume-based scores
        self.latentSpace_slice.extend(latentSpace)
        self.eval_dict['latentSpace'].append(torch.mean(torch.stack(latentSpace),0))
        AnomalyScoreComb_vol = np.mean(AnomalyScoreComb) 
        AnomalyScoreReg_vol = np.mean(AnomalyScoreReg)
        AnomalyScoreReco_vol = np.mean(AnomalyScoreReco)

        self.eval_dict['AnomalyScoreRegPerVol'].append(AnomalyScoreReg_vol)


        if not self.cfg.get('use_postprocessed_score', True):
            self.eval_dict['AnomalyScoreRecoPerVol'].append(AnomalyScoreReco_vol)
            self.eval_dict['AnomalyScoreCombPerVol'].append(AnomalyScoreComb_vol)
            self.eval_dict['AnomalyScoreCombiPerVol'].append(AnomalyScoreReco_vol * AnomalyScoreReg_vol)
            self.eval_dict['AnomalyScoreCombPriorPerVol'].append(AnomalyScoreReco_vol + self.cfg.beta * 0)
            self.eval_dict['AnomalyScoreCombiPriorPerVol'].append(AnomalyScoreReco_vol * 0)

        final_volume = final_volume.unsqueeze(0)
        final_volume = final_volume.unsqueeze(0)
        # calculate metrics

        _test_step(self, final_volume, data_orig, data_seg, data_mask, batch_idx, ID, label) # everything that is independent of the model choice

           
    def on_test_end(self) :
        # calculate metrics
        _test_end(self) # everything that is independent of the model choice 

    
    def on_predict_start(self):
        # Init dict
        self.bboxes_dict = dict()

    # NOTE: Should be called when evaluation NOVA dataset
    def predict_step(self, batch, batch_idx):
        latentSpace = []

        input = batch["image"]
        ID = batch["ID"]
        final_volume = torch.zeros([input.size(2), input.size(3), input.size(1)], device = self.device)

        # calc features for guidance
        if self.cfg.get("encoder_mode") == "end2end":
            _, features = self(batch)
        else:
            features = self(input)

        features_single = features

        if self.cfg.condition:
            latentSpace.append(features_single.mean(0).squeeze().detach().cpu())
        else: 
            latentSpace.append(torch.tensor([0],dtype=float).repeat(input.shape[0]))

        if self.cfg.get('noise_ensemble',False): # evaluate with different noise levels
            timesteps = self.cfg.get('step_ensemble',[250,500,750]) # timesteps to evaluate
            reco_ensemble = torch.zeros_like(input)
            for t in timesteps:
                # generate noise
                if self.cfg.get('noisetype') is not None:
                    noise = gen_noise(self.cfg, input.shape).to(self.device)
                else: 
                    noise = None
                loss_diff, reco = self.diffusion(input,cond=features,t=t-1,noise=noise)
                reco_ensemble += reco
                
            reco = reco_ensemble / len(timesteps) # average over timesteps
        else :
            if self.cfg.get('noisetype') is not None:
                noise = gen_noise(self.cfg, input.shape).to(self.device)
            else: 
                noise = None
            loss_diff, reco = self.diffusion(input,cond=features,t=self.test_timesteps-1,noise=noise)

         # reassamble the reconstruction volume
        final_volume = reco.clone().squeeze(0)
        final_volume = final_volume.permute(1,2,0) # to HxWxD

        _predict_step(self, final_volume, input, batch_idx, ID)

    def on_predict_end(self):
        # Save dict into a numpy file
        np.save(f"NOVA_bboxes_dict_fold{self.fold}.npy", self.bboxes_dict)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.cfg.lr)
    
    def update_prefix(self, prefix):
        self.prefix = prefix 
# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from pprint import pformat
from typing import List

import numpy as np
import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_STD, IMAGENET_DEFAULT_MEAN
from timm.models.layers import trunc_normal_

import src.models.modules.spark.encoder as encoder
from src.models.modules.spark.decoder import LightDecoder
from src.models.modules.spark.models import build_sparse_encoder, build_encoder

class SparK_2D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        mask_ratio=cfg.get('mask_ratio',0.6)
        mask_ratio2=cfg.get('mask_ratio2',mask_ratio)
        uniform=cfg.get('uniform',False)
        using_pe=cfg.get('pe',False)
        pix_norm=cfg.get('pix_norm',1)
        dense_loss=cfg.get('dense_loss',False) 
        loss_l2=cfg.get('loss_l2',True)
        en_de_norm='bn' 
        en_de_lin=True
        sbn=False
        pyramid=cfg.get('pyramid',4)  # 1 for single-scale pre-training; 4 for full-scale pre-training
        self.cfg = cfg
        sparse_encoder = build_sparse_encoder(cfg.backbone, 
                                                input_size=int(cfg.imageDim[1]/cfg.rescaleFactor), 
                                                sbn=sbn, 
                                                drop_path_rate=cfg.get('dp',0), 
                                                verbose=False)
        dense_decoder = LightDecoder(cfg.get('dec_dim',512), 
                                    sparse_encoder.downsample_raito,
                                    double=cfg.get('double',True), 
                                    heavy=cfg.get('hea',[0,1]), 
                                    cmid=cfg.get('cmid',0), 
                                    sbn=sbn)
        input_size, downsample_raito = sparse_encoder.input_size, sparse_encoder.downsample_raito
        self.downsample_raito = downsample_raito
        fmap_size = input_size // downsample_raito
        self.fmap_size = fmap_size
        if mask_ratio != mask_ratio2 and not uniform:                           # with an extra active site
            k = 1 / fmap_size**2
            mask_ratio = min(1, mask_ratio / (1-k))
            mask_ratio2 = min(1, mask_ratio2 / (1-k))
        self.mask_ratio = (mask_ratio, mask_ratio2)
        self.ratios = torch.tensor([self.mask_ratio[0], self.mask_ratio[1], (self.mask_ratio[0] + self.mask_ratio[1]) / 2])
        self.uniform = uniform
        self.len_keep = round(fmap_size * fmap_size * (1-mask_ratio))
        self.pix_norm = int(pix_norm)
        
        self.sparse_encoder = sparse_encoder
        self.dense_decoder = dense_decoder
        
        self.sbn = sbn
        self.pyramid = pyramid
        en_de_norm = en_de_norm.lower()
        self.en_de_norm_str = en_de_norm

        self.en_de_lin_bool = en_de_lin
        self.en_de_norms = nn.ModuleList()
        self.en_de_lins = nn.ModuleList()
        self.using_pe = using_pe
        self.pos_embeds = nn.ParameterList()
        self.mask_tokens = nn.ParameterList()

        fea, d_fea, fmap = self.sparse_encoder.fea_dim, self.dense_decoder.fea_dim, fmap_size
        for i in range(self.pyramid):
            if en_de_norm == 'bn':
                n = (encoder.SparseSyncBatchNorm2d if sbn else encoder.SparseBatchNorm2d)(fea)
            elif en_de_norm == 'ln':
                n = encoder.SparseConvNeXtLayerNorm(fea, data_format='channels_first', sparse=True)
            else:
                n = nn.Identity()
            self.en_de_norms.append(n)
            
            kernel_size = 1 if i <= 0 else 3
            l = nn.Conv2d(fea, d_fea, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=True)
            print(f'[mid, py={self.pyramid}][edl {i}]: k={kernel_size}, #para = {sum(x.numel() for x in l.parameters())/1e6:.2f}')
            if i == 0 and fea == d_fea:
                l = nn.Identity()
            self.en_de_lins.append(l)
        
            if self.using_pe:
                p = torch.from_numpy(get_2d_sincos_pos_embed(fea, fmap)).float()
                p = p.reshape(1, fmap, fmap, fea).permute(0, 3, 1, 2).contiguous()
                p = nn.Parameter(p, requires_grad=False)
                self.pos_embeds.append(p)
            
            p = nn.Parameter(torch.zeros(1, fea, 1, 1))
            trunc_normal_(p, mean=0, std=.02, a=-.02, b=.02)
            self.mask_tokens.append(p)
            fea //= 2
            d_fea //= 2
            fmap *= 2
        
        print(f'[mid, py={self.pyramid}][mask_tokens]: {tuple(p.numel() for p in self.mask_tokens)}')
        
        self.loss_l2, self.dense_loss = loss_l2, dense_loss

        m = torch.tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1)
        s = torch.tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1)
        self.register_buffer('imn_m', m)
        self.register_buffer('imn_s', s)
        # self.register_buffer('norm_black', (torch.ones(1, 3, input_size, input_size) * 0.45 - m) / s)
        self.register_buffer('norm_black', torch.zeros(1, 3, input_size, input_size))
        self.vis_active = self.vis_active_ex = self.vis_inp = self.vis_inp_mask = ...

        self.masking_mode = 'original' # 'original', 'variable_size', 'superpixel'

    

    def visualize_input_and_masked(self, inp_bchw, masked_bchw, idx=0, save_path=None):
        """
        Visualiza a imagem original e a imagem mascarada lado a lado.

        Args:
            inp_bchw: Tensor (B, C, H, W)
            masked_bchw: Tensor (B, C, H, W)
            idx: índice do item no batch a visualizar
            save_path: caminho para guardar o .png (opcional)
        """
        import matplotlib.pyplot as plt

        img_orig = inp_bchw[idx].detach().cpu().permute(1, 2, 0)
        img_mask = masked_bchw[idx].detach().cpu().permute(1, 2, 0)

        is_grayscale = img_orig.shape[-1] == 1

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(img_orig.squeeze(-1) if is_grayscale else img_orig, cmap='gray')
        ax1.set_title("Input Image")
        ax1.axis('off')

        ax2.imshow(img_mask.squeeze(-1) if is_grayscale else img_mask, cmap='gray')
        ax2.set_title("Masked Image")
        ax2.axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    
    def mask(self, shape, device, generator=None, mode='original'):
        """
        Generates binary masks for the input tensor.
        """

        if mode == 'original':
            B, C, H, W = shape
            f = self.fmap_size
            if self.mask_ratio[0] == self.mask_ratio[1]: # fixed mask ratio
                len_keep = self.len_keep
            elif self.uniform:
                r = random.uniform(self.mask_ratio[0], self.mask_ratio[1])
                len_keep = round(f * f * (1-r))
            else:
                i1, i2, i3, i4 = np.linspace(0, B, 4, dtype=int).tolist()
                l1, l2, l3 = i2-i1, i3-i2, i4-i3
                r1, r2, r3 = self.ratios[torch.randperm(3, generator=generator)].tolist()
                r = torch.tensor([r1]*l1 + [r2]*l2 + [r3]*l3, device=device).view(-1, 1, 1)
                active = torch.rand(B, f, f, device=device, generator=generator) >= r

                rr, cc = torch.randint(low=0, high=f, size=(2, B), generator=generator).unbind(0)
                active[torch.arange(B), rr, cc] = True   # an extra active site
                return active.unsqueeze_(1)
            
            idx = torch.rand(B, f*f, generator=generator).argsort(dim=1)    #  (B, f*f)
            idx = idx[:, :len_keep].to(device)   # (B, len_keep)
            return torch.zeros(B, f*f, dtype=torch.bool, device=device).scatter_(dim=1, index=idx, value=True).view(B, 1, f, f)
        elif mode == 'variable_size':
            """
            Variable-size patch masking using rectangles approximately 1/10 of image size.
            Returns: (B, 1, H, W) mask where True = unmasked, False = masked
            """
            B, C, H, W = shape

            mask = torch.zeros((B, 1, H, W), dtype=torch.bool, device=device)

            # Define min/max size of patches as fractions of image size
            min_frac, max_frac = 0.05, 0.15  # patches between 5% and 15% of image dims

            for b in range(B):
                num_regions = torch.randint(1, 1, (1,), generator=generator).item()
                for _ in range(num_regions):
                    h_frac = torch.rand(1, generator=generator).uniform_(min_frac, max_frac).item()
                    w_frac = torch.rand(1, generator=generator).uniform_(min_frac, max_frac).item()
                    h_size = int(H * h_frac)
                    w_size = int(W * w_frac)

                    y = torch.randint(0, H - h_size + 1, (1,), generator=generator).item()
                    x = torch.randint(0, W - w_size + 1, (1,), generator=generator).item()

                    mask[b, 0, y:y + h_size, x:x + w_size] = False  # mark region as masked

            return mask
        elif mode == 'superpixel':
            """
            Simulated superpixel masking on patch grid (f x f).
            Generates blob-like masks via region growing from seeds.
            """
            B, C, H, W = shape
            f = self.fmap_size
            mask = torch.zeros((B, f, f), dtype=torch.bool, device=device)

            def grow_region(mask_b, seed_y, seed_x, size):
                frontier = [(seed_y, seed_x)]
                visited = set()
                for _ in range(size):
                    if not frontier:
                        break
                    y, x = frontier.pop()
                    if (y, x) in visited or not (0 <= y < f and 0 <= x < f):
                        continue
                    visited.add((y, x))
                    mask_b[y, x] = True
                    neighbors = [(y+dy, x+dx) for dy in [-1, 0, 1] for dx in [-1, 0, 1]]
                    frontier.extend(neighbors)

            for b in range(B):
                num_regions = torch.randint(2, 5, (1,), generator=generator).item()
                for _ in range(num_regions):
                    seed_y = torch.randint(0, f, (1,), generator=generator).item()
                    seed_x = torch.randint(0, f, (1,), generator=generator).item()
                    size = torch.randint(5, 20, (1,), generator=generator).item()
                    grow_region(mask[b], seed_y, seed_x, size)

            return mask.unsqueeze(1)  # shape: (B, 1, f, f)
        else:
            raise NotImplementedError(f'[SparK_2D.mask] mode={mode} not implemented yet!')
    
    def forward(self, raw_inp: torch.Tensor, active=None):

        if self.cfg.get("encoder_mode") == "end2end":
            inp_bchw = raw_inp["orig"]
            raw_inp = raw_inp["orig"]
        else:
            inp_bchw = raw_inp
        
        # spatial mask
        if active is None:
            # NOTE: Added mode 'variable_size' for variable-size patch masking and 'superpixel' for simulated superpixel masking. Default is 'original'.
            active: torch.BoolTensor = self.mask(inp_bchw.shape, inp_bchw.device, mode=self.masking_mode)   # (B, 1, f, f)
        encoder._cur_active = active
        if self.masking_mode == "original":
            active_ex = active.repeat_interleave(self.downsample_raito, 2).repeat_interleave(self.downsample_raito, 3)    # (B, 1, H, W)
        else:
            active_ex = active
        masked_bchw = inp_bchw * active_ex

        # Visualize input and masked input
        # for i in range(5):
        #    self.visualize_input_and_masked(inp_bchw, masked_bchw, idx=i, save_path=f"/home/cristiano/phd/conditioned-diffusion-models-UAD/src_hssh/src/models/modules/spark/spark_masked_input_{i}.png")

        # get hierarchical encoded sparse features (a list containing four feature maps)
        fea_bcffs: List[torch.Tensor] = self.sparse_encoder(masked_bchw, pyramid=self.pyramid)
        fea_bcffs.reverse()                     # from the smallest feature map to the largest

        if self.cfg.get("encoder_mode") == "end2end":
            fea_orig: List[torch.Tensor] = self.sparse_encoder(inp_bchw, pyramid=self.pyramid)
            fea_orig.reverse()
        
        cur_active = active
        to_dec = []
        for i, bcff in enumerate(fea_bcffs):    # from the smallest feature map to the largest
            if bcff is not None:
                # fill in empty positions with [mask] embeddings
                bcff = self.en_de_norms[i](bcff)
                mask_tokens = self.mask_tokens[i].expand_as(bcff).type_as(bcff)
                if self.using_pe:
                    mask_tokens = mask_tokens + self.pos_embeds[i].expand_as(bcff)
                bcff = torch.where(cur_active.expand_as(bcff), bcff, mask_tokens)
                bcff: torch.Tensor = self.en_de_lins[i](bcff)
            to_dec.append(bcff)
            cur_active = cur_active.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)     # dilate the mask map
        
        # decode and reconstruct
        rec_bchw = self.dense_decoder(to_dec)
        
        # calc loss
        mean, var, spatial_loss = self.spatial_loss(raw_inp, rec_bchw, active)

        if self.cfg.get("encoder_mode") == "end2end":
            return active_ex, rec_bchw, spatial_loss, fea_bcffs, fea_orig
        else:
            return active_ex, rec_bchw, spatial_loss, fea_bcffs

    def spatial_loss(self, inp, rec, active):   # active: (B, 1, f, f)
        mean = var = None
        if self.pix_norm == 2:
            mean, var = inp.mean(dim=(2, 3), keepdim=True), None
            rec = rec + mean
        inp = self.patchify(inp)
        rec = self.patchify(rec)
        # (B, L=fmap_size**2, N=downsample_raito**2 * C)
        if self.pix_norm == 1:
            mean = inp.mean(dim=-1, keepdim=True)
            var = (inp.var(dim=-1, keepdim=True) + 1e-6)**.5
            inp = (inp - mean) / var
        loss_spa = (rec-inp)**2 if self.loss_l2 else (rec-inp).abs()
        
        if self.dense_loss:
            return mean, var, loss_spa.mean()  # mean loss on all patches
        else:
            loss_spa = loss_spa.mean(dim=2, keepdim=False)  # (B, L, C) => (B, L)
            non_active = active.logical_not().int().view(active.shape[0], -1)  # (B, 1, f, f) => (B, L)
            return mean, var, loss_spa.mul_(non_active).sum() / (non_active.sum() + 1e-8)  # mean loss on removed patches
        
    def patchify(self, bchw):
        p = self.downsample_raito
        h = w = self.fmap_size
        B, C = bchw.shape[:2]
        bchw = bchw.reshape(shape=(B, C, h, p, w, p))
        bchw = torch.einsum('bchpwq->bhwpqc', bchw)
        bln = bchw.reshape(shape=(B, h*w, p**2 * C))    # (B, f*f, downsample_raito*downsample_raito*3)
        return bln
    
    def unpatchify(self, bln):
        p = self.downsample_raito
        h = w = self.fmap_size
        B, C = bln.shape[0], bln.shape[-1] // p**2
        bln = bln.reshape(shape=(B, h, w, p, p, C))
        bln = torch.einsum('bhwpqc->bchpwq', bln)
        bchw = bln.reshape(shape=(B, C, h * p, w * p))
        return bchw

    def denorm_for_vis(self, x, clamp):
        x = x * self.imn_s
        x += self.imn_m
        if clamp:
            x = torch.clamp(x, 0, 1)
        return x

    def __repr__(self):
        return (
            f'\n'
            f'[SparK.config]: {pformat(self.get_config(), indent=2, width=250)}\n'
            f'[SparK.structure]: {super(SparK, self).__repr__().replace(SparK.__name__, "")}\n'
            f'[SparK.dec]: {self.dense_decoder.num_para()}'
        )

    def get_config(self):
        return {
            # self
            'mask_ratio': self.mask_ratio[0], 'mask_ratio2': self.mask_ratio[1], 'uniform': self.uniform,
            'using_pe': self.using_pe, 'pix_norm': self.pix_norm,
            'dense_loss': self.dense_loss, 'loss_l2': self.loss_l2,
            'en_de_norm': self.en_de_norm_str, 'en_de_lin': self.en_de_lin_bool, 'sbn': self.sbn, 'pyramid': self.pyramid,
    
            # enc
            'input_size': self.sparse_encoder.input_size,
            # dec
            'dec_fea_dim': self.dense_decoder.fea_dim, 'double': self.dense_decoder.double_bool, 'heavy': self.dense_decoder.heavy,
        }
    
    def state_dict(self, destination=None, prefix='', keep_vars=False, with_config=False):
        state = super(SparK_2D, self).state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        if with_config:
            state['config'] = self.get_config() # todo: 似乎会引起DDP broadcast err？？
        return state
    
    def load_state_dict(self, state_dict, strict=True):
        config = state_dict.pop('config', None)
        incompatible_keys = super(SparK_2D, self).load_state_dict(state_dict, strict=strict)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[SparseMIM.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err)
                
        return incompatible_keys

class SparK_2D_encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        sbn=False
        self.pyramid=cfg.get('pyramid',4)  # 1 for single-scale pre-training; 4 for full-scale pre-training
        self.cfg = cfg
        backbone = cfg.version # resnet50 convnext_base 

        self.encoder = build_encoder(backbone, 
                                    cfg.get('cond_dim',128),
                                    input_size=int(cfg.imageDim[1]/cfg.rescaleFactor), 
                                    sbn=sbn, 
                                    drop_path_rate=cfg.get('dp',0), 
                                    verbose=False)
        

    def forward(self, x):
        features = self.encoder(x)
        return features

def _make_divisible(v, divisor=8, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    
    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)
    
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product
    
    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)
    
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


if __name__ == '__main__':
    SparK_2D.test_mask()
    SparK_2D.test_align()
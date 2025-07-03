import timm 
import torch 
import torchvision
from src.models.modules.spark.Spark_2D import SparK_2D_encoder
from src.models.modules.contrastive.Contrastive_Encoder import LoadContrastiveModel
from src.models.modules.mi2.MI2_Encoder import LoadMI2Model

def get_encoder(cfg, fold):
    """
    Available backbones (some of them): 
    Resnet: 
        resnet18,
        resnet34,
        resnet50, 
        resnet101
    """
    backbone = cfg.get('backbone','resnet50')  
    chans = 1 
    if 'spark' in backbone.lower(): # spark encoder
        encoder = SparK_2D_encoder(cfg)
    elif 'contrastive' in backbone.lower(): # contrastive encoder
        encoder = LoadContrastiveModel(cfg, fold)
    elif 'mi2' in backbone.lower(): # MI2 encoder
        encoder = LoadMI2Model(cfg, fold)
    else : # 2D CNN encoder
        encoder = timm.create_model(backbone, pretrained=cfg.pretrained_backbone, in_chans=chans, num_classes = cfg.get('cond_dim',256) )
                               
    out_features = cfg.get('cond_dim',256) 
    
    return encoder, out_features
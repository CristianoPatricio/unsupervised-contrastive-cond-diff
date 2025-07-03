from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
from typing import Optional
import pandas as pd
import src.datamodules.create_dataset as create_dataset
from src.datamodules.create_dataset import NOVADataset


class Brats21(LightningDataModule):

    def __init__(self, cfg, fold= None):
        super(Brats21, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload',True)
        # load data paths and indices
        self.imgpath = {}
        self.csvpath_val = cfg.path.Brats21.IDs.val
        self.csvpath_test = cfg.path.Brats21.IDs.test
        self.csv = {}
        states = ['val','test']

        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)
        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'Brats21'

            self.csv[state]['img_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['img_path']
            self.csv[state]['mask_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['mask_path']
            self.csv[state]['seg_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['seg_path']

            if cfg.mode != 't1':
                self.csv[state]['img_path'] = self.csv[state]['img_path'].str.replace('t1',cfg.mode).str.replace('FLAIR.nii.gz',f'{cfg.mode.lower()}.nii.gz')

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        if not hasattr(self,'val_eval'):
            if self.cfg.sample_set: # for debugging
                self.val_eval = create_dataset.Eval(self.csv['val'][0:8], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:8], self.cfg)
            else :
                self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)

    def val_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)



class MSLUB(LightningDataModule):

    def __init__(self, cfg, fold= None):
        super(MSLUB, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload',True)
        # load data paths and indices
        self.imgpath = {}
        self.csvpath_val = cfg.path.MSLUB.IDs.val
        self.csvpath_test = cfg.path.MSLUB.IDs.test
        self.csv = {}
        states = ['val','test']

        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)
        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'MSLUB'

            self.csv[state]['img_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['img_path']
            self.csv[state]['mask_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['mask_path']
            self.csv[state]['seg_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['seg_path']
            
            if cfg.mode != 't1':
                #self.csv[state]['img_path'] = self.csv[state]['img_path'].str.replace('uniso/t1',f'uniso/{cfg.mode}').str.replace('t1.nii.gz',f'{cfg.mode}.nii.gz')
                self.csv[state]['img_path'] = self.csv[state]['img_path'].str.replace('MSLUB/t1',f'MSLUB/{cfg.mode}').str.replace('t1.nii.gz',f'{cfg.mode}.nii.gz')
    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        if not hasattr(self,'val_eval'):
            if self.cfg.sample_set: # for debugging
                self.val_eval = create_dataset.Eval(self.csv['val'][0:4], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:4], self.cfg)
            else :
                self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)

    def val_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)
    

# NOTE: Added here to handle NOVA dataset
class NOVA(LightningDataModule):
    """
    Class that handle the loading of NOVA dataset.
    """
    def __init__(self, cfg, fold=None):
        super(NOVA, self).__init__()
        self.cfg = cfg
        self.csv_path = cfg.path.NOVA.csv_path
        self.image_dir = cfg.path.NOVA.image_dir

    def setup(self, stage: Optional[str] = None):
        self.test_dataset = NOVADataset(config=self.cfg, csv_path=self.csv_path, image_dir=self.image_dir)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)


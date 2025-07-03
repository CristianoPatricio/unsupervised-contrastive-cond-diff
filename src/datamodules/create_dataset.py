from torch.utils.data import Dataset
import numpy as np
import torch
import SimpleITK as sitk
import torchio as tio
from torch import Tensor
from torchvision import transforms
from torchvision.transforms.functional import erase
sitk.ProcessObject.SetGlobalDefaultThreader("Platform")
from typing import Any, Callable, List, Optional, Tuple, Union
from multiprocessing import Manager
import pandas as pd
import os
from PIL import Image
import ast

def Train(csv,cfg,preload=True):
    subjects = []
    for _, sub in csv.iterrows():
        subject_dict = {
            'vol' : tio.ScalarImage(sub.img_path,reader=sitk_reader), 
            'age' : sub.age,
            'ID' : sub.img_name,
            'label' : sub.label,
            'Dataset' : sub.setname,
            'stage' : sub.settype,
            'path' : sub.img_path
        }
        if sub.mask_path != None: # if we have masks
            subject_dict['mask'] = tio.LabelMap(sub.mask_path,reader=sitk_reader)
        else: # if we don't have masks, we create a mask from the image
            subject_dict['mask'] = tio.LabelMap(tensor=tio.ScalarImage(sub.img_path,reader=sitk_reader).data>0)

        subject = tio.Subject(subject_dict)
        subjects.append(subject)
    
    if preload: 
        manager = Manager()
        cache = DatasetCache(manager)
        ds = tio.SubjectsDataset(subjects, transform = get_transform(cfg))
        ds = preload_wrapper(ds, cache, augment = get_augment(cfg))
    else: 
        ds = tio.SubjectsDataset(subjects, transform = tio.Compose([get_transform(cfg),get_augment(cfg)]))
        
    if cfg.spatialDims == '2D':
        slice_ind = cfg.get('startslice',None) 
        seq_slices = cfg.get('sequentialslices',None) 
        ds = vol2slice(ds,cfg,slice=slice_ind,seq_slices=seq_slices)
    return ds
 
def Eval(csv,cfg): 
    subjects = []
    for _, sub in csv.iterrows():
        if sub.mask_path is not None and tio.ScalarImage(sub.img_path,reader=sitk_reader).shape != tio.ScalarImage(sub.mask_path,reader=sitk_reader).shape:
            print(f'different shapes of vol and mask detected. Shape vol: {tio.ScalarImage(sub.img_path,reader=sitk_reader).shape}, shape mask: {tio.ScalarImage(sub.mask_path,reader=sitk_reader).shape} \nsamples will be resampled to the same dimension')
            
        subject_dict = {
            'vol' : tio.ScalarImage(sub.img_path,reader=sitk_reader),
            'vol_orig' : tio.ScalarImage(sub.img_path,reader=sitk_reader), # we need the image in original size for evaluation
            'age' : sub.age,
            'ID' : sub.img_name,
            'label' : sub.label,
            'Dataset' : sub.setname,
            'stage' : sub.settype,
            'seg_available': False,
            'path' : sub.img_path }
        if sub.seg_path != None: # if we have segmentations
            subject_dict['seg'] = tio.LabelMap(sub.seg_path,reader=sitk_reader),
            subject_dict['seg_orig'] = tio.LabelMap(sub.seg_path,reader=sitk_reader)# we need the image in original size for evaluation
            subject_dict['seg_available'] = True
        if sub.mask_path != None: # if we have masks
            subject_dict['mask'] = tio.LabelMap(sub.mask_path,reader=sitk_reader)
            subject_dict['mask_orig'] = tio.LabelMap(sub.mask_path,reader=sitk_reader)# we need the image in original size for evaluation
        else: 
            tens=tio.ScalarImage(sub.img_path,reader=sitk_reader).data>0
            subject_dict['mask'] = tio.LabelMap(tensor=tens)
            subject_dict['mask_orig'] = tio.LabelMap(tensor=tens)

        subject = tio.Subject(subject_dict)
        subjects.append(subject)
    ds = tio.SubjectsDataset(subjects, transform = get_transform(cfg))
    return ds
## got it from https://discuss.pytorch.org/t/best-practice-to-cache-the-entire-dataset-during-first-epoch/19608/12
class DatasetCache(object):
    def __init__(self, manager, use_cache=True):
        self.use_cache = use_cache
        self.manager = manager
        self._dict = manager.dict()

    def is_cached(self, key):
        if not self.use_cache:
            return False
        return str(key) in self._dict

    def reset(self):
        self._dict.clear()

    def get(self, key):
        if not self.use_cache:
            raise AttributeError('Data caching is disabled and get funciton is unavailable! Check your config.')
        return self._dict[str(key)]

    def cache(self, key, subject):
        # only store if full data in memory is enabled
        if not self.use_cache:
            return
        # only store if not already cached
        if str(key) in self._dict:
            return
        self._dict[str(key)] = (subject)

class preload_wrapper(Dataset):
    def __init__(self,ds,cache,augment=None):
            self.cache = cache
            self.ds = ds
            self.augment = augment
    def reset_memory(self):
        self.cache.reset()
    def __len__(self):
            return len(self.ds)
            
    def __getitem__(self, index):
        if self.cache.is_cached(index) :
            subject = self.cache.get(index)
        else:
            subject = self.ds.__getitem__(index)
            self.cache.cache(index, subject)
        if self.augment:
            subject = self.augment(subject)
        return subject
    
# Customized Transformation
class RandomErasingBrain(torch.nn.Module):
    """Randomly selects a rectangle region in a torch.Tensor image and erases its pixels.
    This transform does not support PIL Image.
    Adapted from RandomErasing PyTorch transform: https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomErasing.html

    Args:
         p: probability that the random erasing operation will be performed.
         b: value controlling the amount of noise added to the transform (b=1 means no noise added).
         ratio: range of aspect ratio of erased area.
         value: erasing value.

    Returns:
        Erased Image.

    Example:
        >>> transform = transforms.Compose([
        >>>   transforms.RandomHorizontalFlip(),
        >>>   transforms.PILToTensor(),
        >>>   transforms.ConvertImageDtype(torch.float),
        >>>   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>>   transforms.RandomErasingBrain(p=0.7, b=0.7, ratio=(0.5, 0.2)),
        >>> ])
    """

    def __init__(self, p=0.5, b=1, ratio=(0.7, 0.2), value=0):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("Random erasing probability should be between 0 and 1")
        if b < 0 or b > 1:
            raise ValueError("Random noise probability should be between 0 and 1")
        if ratio[0] < 0 or ratio[1] > 1:
            raise ValueError("Ratio should be between 0 and 1")

        self.p = p
        self.b = b
        self.ratio = ratio
        self.value = value

    @staticmethod
    def get_params(
            img: Tensor, b: float, ratio: Tuple[float, float], value: Optional[List[float]] = None
    ) -> Tuple[int, int, int, int, Tensor]:
        """Get parameters for ``erase`` for a random erasing.

        Args:
            img (Tensor): Tensor image to be erased.
            b (float): parameter which controls the bias of the augmentation
            ratio (sequence): range of aspect ratio of erased area (% of width, % of height).
            value (list, optional): erasing value. If None, it is interpreted as "random"
                (erasing each pixel with random values). If ``len(value)`` is 1, it is interpreted as a number,
                i.e. ``value[0]``.

        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        img_c, img_h, img_w = img.shape[-3], img.shape[-2], img.shape[-1]

        factor = torch.normal(mean=0.0, std=1.0 - b, size=(1, 1))

        min = -2.0  # (X ~ N(0;1) -> P(X < -2) = 0.00227)
        max = 2.0  # (X ~ N(0;1) -> P(X > 2) = 0.00227)

        # Normalize the amount of noise added between the specified range
        w_norm = (0.6 * ((factor - min) / (max - min))) - 0.3 if b != 1.0 else 0.0  # we want values from -0.3 to 0.3
        h_norm = (0.3 * ((factor - min) / (max - min))) - 0.1 if b != 1.0 else 0.0  # we want values from -0.1 a 0.2

        h = int(img_h * (ratio[1] + h_norm))  # h: [0.1, 0.4] of img height
        w = int(img_w * (ratio[0] + w_norm))  # w: [0.2, 0.8] of img width

        v = torch.tensor(value).item()

        i = int((int((img_h - h) / 2) + int(
            factor * (img_h / 2))) % img_h)  # change y coordinate based on the sampled factor
        j = int((int((img_w - w) / 2) + int(
            factor * (img_w / 2))) % img_w)  # change x coordinate based on the sampled factor

        return i, j, h, w, v

    def forward(self, img):
        """
        Args:
            img (Tensor): Tensor image to be erased.

        Returns:
            img (Tensor): Erased Tensor image.
        """
        if torch.rand(1) < self.p:
            x, y, h, w, v = self.get_params(img, b=self.b, ratio=self.ratio, value=self.value)
            # print(f"x:{x}, y:{y}, h:{h}, w:{w}, v:{v}")
            return erase(img, x, y, h, w, v)
        return img

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}"
            f"(p={self.p}, "
            f"b={self.b}, "
            f"ratio={self.ratio}, "
            f"value={self.value})"
        )
        return s

class vol2slice(Dataset):
    def __init__(self,ds,cfg,onlyBrain=False,slice=None,seq_slices=None):
            self.ds = ds
            self.onlyBrain = onlyBrain
            self.slice = slice
            self.seq_slices = seq_slices
            self.counter = 0 
            self.ind = None
            self.cfg = cfg

    def __len__(self):
            return len(self.ds)
            
    def __getitem__(self, index):
        subject = self.ds.__getitem__(index)
        if self.onlyBrain:
            start_ind = None
            for i in range(subject['vol'].data.shape[-1]):
                if subject['mask'].data[0,:,:,i].any() and start_ind is None: # only do this once
                    start_ind = i 
                if not subject['mask'].data[0,:,:,i].any() and start_ind is not None: # only do this when start_ind is set
                    stop_ind = i 
            low = start_ind
            high = stop_ind
        else: 
            low = 0
            high = subject['vol'].data.shape[-1]
        if self.slice is not None:
            self.ind = self.slice
            if self.seq_slices is not None:
                low = self.ind
                high = self.ind + self.seq_slices
                self.ind = torch.randint(low,high,size=[1])
        else:
            if self.cfg.get('unique_slice',False): # if all slices in one batch need to be at the same location
                if self.counter % self.cfg.batch_size == 0 or self.ind is None: # only change the index when changing to new batch
                    self.ind = torch.randint(low,high,size=[1])
                self.counter = self.counter +1
            else: 
                self.ind = torch.randint(low,high,size=[1])

        subject['ind'] = self.ind

        subject['vol'].data = subject['vol'].data[...,self.ind]
        subject['mask'].data = subject['mask'].data[...,self.ind]

        if self.cfg.get('encoder_mode') == "end2end":
            # Save only vol data
            subject_orig = subject['vol'].data.squeeze(-1)

            subject['orig'] = subject_orig

            # Define custom transform to apply RandomErasingFaces
            custom_transform = transforms.Compose([
                transforms.ToTensor(),
                RandomErasingBrain(p=self.cfg.re_prob, b=0.7, ratio=(0.5, 0.2))
            ])

            # Apply transform to the image
            subject_aug = custom_transform(subject['vol'].data.numpy().squeeze(0))

            # Convert to RGB
            if self.cfg.encoder_model == "mi2":
                subject_orig = np.repeat(subject_orig, 3, axis=0)
                subject_aug = np.repeat(subject_aug, 3, axis=0)

            subject['augmented'] = subject_aug
 
            # NOTE: Uncomment below for debugging
            #print(f"Shape of the original volume: {subject_orig.shape}")
            #print(f"Shape of the image before augmentation: {subject['vol'].data.numpy().squeeze(0).shape}")
            #print(f"Shape of the image after the augmentation: {subject_aug.shape}")
            #save_image(subject_orig, "/scratch/orig.png")
            #save_image(subject_aug, "/scratch/aug.png")

        return subject


def get_transform(cfg): # only transforms that are applied once before preloading
    h, w, d = tuple(cfg.get('imageDim',(160,192,160)))

    if not cfg.resizedEvaluation: 
        exclude_from_resampling = ['vol_orig','mask_orig','seg_orig']
    else: 
        exclude_from_resampling = None
        
    if cfg.get('unisotropic_sampling',True):
        preprocess = tio.Compose([
        tio.CropOrPad((h,w,d),padding_mode=0),
        tio.RescaleIntensity((0, 1),percentiles=(cfg.get('perc_low',1),cfg.get('perc_high',99)),masking_method='mask'),
        tio.Resample(cfg.get('rescaleFactor',3.0),image_interpolation='bspline',exclude=exclude_from_resampling),#,exclude=['vol_orig','mask_orig','seg_orig']), # we do not want to resize *_orig volumes
        ])

    else: 
        preprocess = tio.Compose([
                tio.RescaleIntensity((0, 1),percentiles=(cfg.get('perc_low',1),cfg.get('perc_high',99)),masking_method='mask'),
                tio.Resample(cfg.get('rescaleFactor',3.0),image_interpolation='bspline',exclude=exclude_from_resampling),#,exclude=['vol_orig','mask_orig','seg_orig']), # we do not want to resize *_orig volumes
            ])


    return preprocess 

def get_augment(cfg): # augmentations that may change every epoch
    augmentations = []

    # individual augmentations
    if cfg.get('random_bias',False):
        augmentations.append(tio.RandomBiasField(p=0.25))
    if cfg.get('random_motion',False):
        augmentations.append(tio.RandomMotion(p=0.1))
    if cfg.get('random_noise',False):
        augmentations.append(tio.RandomNoise(p=0.5))
    if cfg.get('random_ghosting',False):
        augmentations.append(tio.RandomGhosting(p=0.5))
    if cfg.get('random_blur',False):
        augmentations.append(tio.RandomBlur(p=0.5))
    if cfg.get('random_gamma',False):        
        augmentations.append(tio.RandomGamma(p=0.5))
    if cfg.get('random_elastic',False):
        augmentations.append(tio.RandomElasticDeformation(p=0.5))
    if cfg.get('random_affine',False):
        augmentations.append(tio.RandomAffine(p=0.5))
    if cfg.get('random_flip',False):
        augmentations.append(tio.RandomFlip(p=0.5))

    # policies/groups of augmentations
    if cfg.get('aug_intensity',False): # augmentations that change the intensity of the image rather than the geometry
        augmentations.append(tio.RandomGamma(p=0.5))
        augmentations.append(tio.RandomBiasField(p=0.25))
        augmentations.append(tio.RandomBlur(p=0.25))
        augmentations.append(tio.RandomGhosting(p=0.5))

    augment = tio.Compose(augmentations)
    return augment
def sitk_reader(path):
                
    image_nii = sitk.ReadImage(str(path), sitk.sitkFloat32)
    if not 'mask' in str(path) and not 'seg' in str(path) : # only for volumes / scalar images
        image_nii = sitk.CurvatureFlow(image1 = image_nii, timeStep = 0.125, numberOfIterations = 3)
    vol = sitk.GetArrayFromImage(image_nii).transpose(2,1,0)
    return vol, None


def get_transform_NOVA(): # only transforms that are applied once before preloading
    import torchvision.transforms as T

    # Transforma para tensor [1, H, W] com valores [0, 1]
    to_tensor = T.Compose([
        T.ToTensor(),  # converte para float32 automaticamente e normaliza para [0, 1]
    ])


    return to_tensor 


# NOTE: Added here to handle NOVA dataset
class NOVADataset(Dataset):
    def __init__(self, config, csv_path, image_dir, transform=None):
        """
        Args:
            csv_path (str): Path to the CSV file.
            image_dir (str): Directory containing image files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.cfg = config
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = get_transform_NOVA()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        image_path = os.path.join(self.image_dir, row["filename"])
        image_2d = Image.open(image_path).convert("L")

        # Apply transforms
        image = self.transform(image_2d) if self.transform else torch.from_numpy(np.array(image_2d))

        sample = {
            "image": image,
            "ID": row["filename"]
        }

        return sample
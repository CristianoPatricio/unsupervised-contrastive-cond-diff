# Unsupervised contrastive analysis for anomaly detection in brain MRIs via conditional diffusion models

Official implementation of the paper [Unsupervised contrastive analysis for anomaly detection in brain MRIs via conditional diffusion models](https://arxiv.org/abs/2406.00772).

## Abstract
Contrastive Analysis (CA) detects anomalies by contrasting patterns unique to a target group (e.g., unhealthy subjects) from those in a background group (e.g., healthy subjects). In the context of brain MRIs, existing CA approaches rely on supervised contrastive learning or variational autoencoders (VAEs) using both healthy and unhealthy data, but such reliance on target samples is challenging in clinical settings. Unsupervised Anomaly Detection (UAD) offers an alternative by learning a reference representation of healthy anatomy without the need for target samples. Deviations from this reference distribution can indicate potential anomalies. In this context, diffusion models have been increasingly adopted in UAD due to their superior performance in image generation compared to VAEs. Nonetheless, precisely reconstructing the anatomy of the brain remains a challenge. In this work, we propose an unsupervised framework to improve the reconstruction quality by training a self-supervised contrastive encoder on healthy images to extract meaningful anatomical features. These features are used to condition a diffusion model to reconstruct the healthy appearance of a given image, enabling interpretable anomaly localization via pixel-wise comparison. We validate our approach through a proof-of-concept on a facial image dataset and further demonstrate its effectiveness on four brain MRI datasets, achieving state-of-the-art anomaly localization performance on the NOVA benchmark.

<p align="center"><img title="Method" width="60%" alt="Unsupervised contrastive analysis for anomaly detection in brain MRIs via conditional diffusion models" src="assets/method.png"></p>

## ⚡️ Highlights 🔥  
- Unsupervised framework enhancing reconstruction quality in brain MRIs;
- Target-invariant contrastive encoder capturing meaningful anatomical features;
- Conditional diffusion model to reconstruct the healthy appearance of a given image;
- Top performance in anomaly localization on the NOVA benchmark.

## Data

The datasets can be downloaded/requested from the following links:

- IXI dataset: https://brain-development.org/ixi-dataset/
- MSLUB dataset: https://lit.fe.uni-lj.si/en/research/resources/3D-MR-MS/
- BraTS2021 dataset: http://braintumorsegmentation.org/
- NOVA dataset: https://huggingface.co/datasets/Ano-2090/Nova

Preprocessed versions of the datasets (IXI, MSLUB and BraTS2021) are available in the [cDDPM](https://github.com/FinnBehrendt/Conditioned-Diffusion-Models-UAD/#Data) repository for download.

After downloading, the directory structure of <DATA_DIR> should look like this: 

    <DATA_DIR>
    ├── Train
    │   └── ixi
    │       ├── mask
    │       └── t2
    ├── Test
    │   ├── Brats21
    │   │   ├── mask
    │   │   ├── t2
    │   │   └── seg
    │   └── MSLUB
    │       ├── mask
    │       ├── t2
    │       └── seg
    ├── splits
    │   ├──  Brats21_test.csv        
    │   ├──  Brats21_val.csv   
    │   ├──  MSLUB_val.csv 
    │   ├──  MSLUB_test.csv
    │   ├──  IXI_train_fold0.csv
    │   ├──  IXI_train_fold1.csv 
    │   └── ...                
    ├── NOVA
    │   ├──  images    
    │   └──  metadata.csv
    └── ...

You should then specify the location of <DATA_DIR> in the pc_environment.env file. Additionally, specify the <LOG_DIR>, where runs will be saved. 

## Environment Setup

Create the conda environment and activate it by running the following commands:

```
conda env create -f environment.yaml
conda activate uad-env
pip install -r requirements.txt
```

## Experiments

1. Target-invariant encoder pretraining
```python
python run.py experiment=contrastive_encoder  
```

2. Train the conditional diffusion model
```python
python run.py experiment=conditional_DDPM encoder_path=<path_to_pretrained_encoder>
```

3. Calculate metrics
```python
python scr_calculate_cond_DDPM_metrics.py --run_path=/path/to/run_dir_of_trained_conditional_DDPM
``` 

4. Get predictions for NOVA benchmark with the best model
```python
python run_NOVA.py experiment=NOVA_benchmark encoder_path=<path_to_pretrained_encoder> load_checkpoint=<path_to_best_ckpt_from_conditional_DDPM>
```

5. Calculate metrics for NOVA benchmark (Modify <PREDS_PATH> in line 178)
```python
python scr_calculate_NOVA_metrics.py
```

## Citation

If you use this repository, please cite:

```
@article{patricio2025UAD,
    title={Unsupervised contrastive analysis for anomaly detection in brain MRIs via conditional diffusion models},
    author = {Cristiano Patrício and Carlo Alberto Barbano and Attilio Fiandrotti and Riccardo Renzulli and Marco Grangetto and Luis F. Teixeira and João C. Neves},
    journal={arXiv preprint arXiv:2406.00772},
    year={2025}
}
```

## Acknowledgment
This codebase is inspired by and built upon the work from [cDDPM](https://github.com/FinnBehrendt/Conditioned-Diffusion-Models-UAD).
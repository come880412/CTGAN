# CTGAN.Pytorch
ICIP 2022: TBD

## Proposed model
### Generator
The overall structure of our proposed CTGAN is illustrated below. We focus more on the design of the feature extractor and processing of the sequential features. We refer to the conformer module, the modified version of Transformer, intending to make the downsampled sequential features find the most critical representation.
<img src="https://github.com/come880412/CTGAN/blob/main/img/Generator.png" width=100% height=100%>

### Feature Extractor
In the feature extractor, we introduce the auxiliary generator and atrous convolution. The former makes the feature extractor converge faster, while the latter enables a larger receptive field in the early stage. In addition, we design a module for detecting the cloud_mask, using it to keep the weight of the cloud-free regions while throwing out the weight of cloudy regions.
<img src="https://github.com/come880412/CTGAN/blob/main/img/Feature_extractor.jpg" width=100% height=100%>

## Visualization
<p align="center">
 <img src="https://github.com/come880412/CTGAN/blob/main/img/visualization.jpg" width=50% height=50%>
</p>

## User instructions
### Computer equipments
- System: Ubuntu20.04
- Python version: Python 3.6 or higher
- Training:\
  CPU: Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz\
  RAM: 256GB\
  GPU: NVIDIA GeForce RTX 3090 24GB\

### Install Packages
Please see the ```requirements.txt ``` for more details.

### Prepare data
#### Sen2_MTC
Please download the dataset from [Sen2_MTC](https://drive.google.com/drive/folders/1xUmr8wTWXPnINKlxr0d0l-q48KAph8EO?usp=sharing).\
The dataset is collected from the public-avalible Sentinel-2 by ourselves. There are 50 non-overlap tiles, each has 70 images with size = (256, 256), channels = 4 (R, G, B, NIR) and pixel value range [0, 10000].

#### STGAN dataset
Please download the dataset from [STGAN dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/BSETKZ).

### Pretrained model
We share two CTGAN pretrained models, one is trained on Sen2_MTC, another is trained on the STGAN dataset.
[CTGAN_Sen2](https://drive.google.com/drive/folders/1-kOSEhogEvmataXAdM3Zq2B_5oPk7tV0?usp=sharing)   [CTGAN_STGAN_dataset](https://drive.google.com/drive/folders/19EiiqATFhJwv19RQszrfcPSh0yXr_GqJ?usp=sharing)

### Inference
```python test.py  --gen_checkpoint_path path/to/model --val_path path/to/val.txt --test_path path/to/test.txt```

### Training
```python train.py --train_path path/to/train.txt --val_path path/to/val.txt --dataset_name Sen2_MTC_CTGAN --batch_size 4```

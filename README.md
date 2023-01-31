# CTGAN.Pytorch (Official)
ICIP 2022: CTGAN : Cloud Transformer Generative Adversarial Network \
[Paper link](https://ieeexplore.ieee.org/document/9897229) \
Authors:
[Gi-Luen Huang](r09942171@ntu.edu.tw), [Pei-Yuan Wu](https://www.ee.ntu.edu.tw/profile1.php?teacher_id=24038)

## Proposed model
### Generator
The overall structure of our proposed CTGAN is illustrated below. We focus more on the design of the feature extractor and processing of the sequential features. We refer to the conformer module, the modified version of Transformer, intending to make the downsampled sequential features find the most critical representation.
<img src="https://github.com/come880412/CTGAN/blob/main/images/Generator.jpg" width=100% height=100%>

### Feature Extractor
In the feature extractor, we introduce the auxiliary generator and atrous convolution. The former makes the feature extractor converge faster, while the latter enables a larger receptive field in the early stage. In addition, we design a module for detecting the cloud_mask, using it to keep the weight of the cloud-free regions while throwing out the weight of cloudy regions.
<img src="https://github.com/come880412/CTGAN/blob/main/images/Feature_extractor.jpg" width=100% height=100%>

## Visualization results
<p align="center">
 <img src="https://github.com/come880412/CTGAN/blob/main/images/visualization.jpg" width=50% height=50%>
</p>

## Getting started
- Clone this repo to your local
``` bash
git clone git@github.com:come880412/CTGAN.git
cd CTGAN
```

### Computer equipments
- System: Ubuntu20.04
- Python version: Python 3.6 or higher
- Training:\
  CPU: Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz\
  RAM: 256GB\
  GPU: NVIDIA GeForce RTX 3090 24GB

### Install Packages
Please see the ```requirements.txt``` for more details.

### Prepare data

Please download the dataset from [Sen2_MTC](https://drive.google.com/file/d/1-hDX9ezWZI2OtiaGbE8RrKJkN1X-ZO1P/view?usp=share_link)
- Sen2_MTC is collected from the public-avalible Sentinel-2 by ourselves. There are 50 non-overlap tiles, each has 70 images with size = (256, 256), channels = 4 (R, G, B, NIR) and pixel value range [0, 10000].
- You can use the python script ```train_val_split.py``` to split the data into train/val/test. Or, you can use the .txt files provided by us to ensure we have the same train/val/test sets.

### Pretrained model
We provide CTGAN pretrained model on the Sen2_MTC dataset. You can download the pretrained models from [here](https://drive.google.com/drive/folders/1-kOSEhogEvmataXAdM3Zq2B_5oPk7tV0?usp=sharing).   

### Inference
- You should first download the pretrained models from [Pretrained model](###Pretrained-model) or train CTGAN by yourself.
``` bash
python test.py  --load_gen path/to/model --root path/to/dataset --test_mode val/test
```

### Training
- You can use the following command to train CTGAN from scratch
``` bash
python train.py --root path/to/dataset --cloud_model_path path/to/Feature_Extrator_FS2.pth --dataset_name Sen2_MTC --batch_size 4 --load_gen '' --load_dis '' 
```
- You can monitor the training process using ```$ tensorboard --logdir=runs``` and then go to the URL [http://localhost:6006/](http://localhost:6006/)
- If you have any implementation problems, please feel free to e-mail me! come880412@gmail.com

### Acknowledgements
Our developed CTGAN was inspired by STGAN ([paper here](https://arxiv.org/abs/1912.06838)) and SPAGAN([paper here](https://arxiv.org/abs/2009.13015)) architectures. Thanks for your contributions to the community.

### Citation
```
@INPROCEEDINGS{9897229,
  author={Huang, Gi-Luen and Wu, Pei-Yuan},
  booktitle={2022 IEEE International Conference on Image Processing (ICIP)}, 
  title={CTGAN : Cloud Transformer Generative Adversarial Network}, 
  year={2022},
  volume={},
  number={},
  pages={511-515},
  doi={10.1109/ICIP46576.2022.9897229}}
```

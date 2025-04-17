[![arXiv](https://img.shields.io/badge/arXiv-2504.11754-b31b1b.svg)](https://arxiv.org/abs/2504.11754)
![code visitors](https://visitor-badge.glitch.me/badge?page_id=vLAR-group/OGC)
[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
[![Twitter Follow](https://img.shields.io/twitter/follow/vLAR_Group?style=social)](https://twitter.com/vLAR_Group)

## GrabS: Generative Embodied Agent for 3D Object Segmentation without Scene Supervision (ICLR 2025 Spotlight)
[Zihui Zhang](https://scholar.google.com.hk/citations?hl=en&user=jiwazT8AAAAJ&view_op=list_works&sortby=pubdate), [Yafei Yang](https://yangyafei1998.github.io/), [Hongtao Wen](https://hatimwen.github.io/), [Bo Yang](https://yang7879.github.io/)

### Overview

We propose an unsupervised framework to separate learning objectness and search objects in 3D scenes.

<img src="figures/overview.jpg" alt="drawing" width=800/>

Our method enables an Agent to search objects in 3D scenes with the aid of Reinforcement Learning:

<img src="figures/01-overview_demo.gif" alt="drawing" width=600/>

## 0. Installation

### Installing dependencies
```shell script
### CUDA 11.3  GCC 9.4
conda env create -f env.yml
# detectron2
pip3 install 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' --no-deps
## Install libopenblas-dev, mink
sudo apt-get install libopenblas-dev
git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
cd MinkowskiEngine
git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
python setup.py install --force_cuda --blas=openblas
#pointnet2
cd ../pointnet2
python setup.py install
```
### Install superpoint dependencies
We also create [SPG](https://github.com/loicland/superpoint_graph) superpoints on S3DIS and the Synthetic data, which are used
to help training. So, please compile the dependencies.
```shell script
conda install -c anaconda boost
conda install -c omnia eigen3
conda install eigen

CONDAENV=YOUR_CONDA_ENVIRONMENT_LOCATION ##e.g. /home/zihui/anaconda3/envs/GOPS
cd partition/ply_c
cmake . -DPYTHON_LIBRARY=$CONDAENV/lib/libpython3.9.so -DPYTHON_INCLUDE_DIR=$CONDAENV/include/python3.9 -DBOOST_INCLUDEDIR=$CONDAENV/include -DEIGEN3_INCLUDE_DIR=$CONDAENV/include/eigen3
make
cd ..
cd cut-pursuit
mkdir build
cd build
cmake .. -DPYTHON_LIBRARY=$CONDAENV/lib/libpython3.9.so -DPYTHON_INCLUDE_DIR=$CONDAENV/include/python3.9 -DBOOST_INCLUDEDIR=$CONDAENV/include -DEIGEN3_INCLUDE_DIR=$CONDAENV/include/eigen3
make
```

[comment]: <> (#### gaps 好像本地不能运行)

## 1. Data Preparation
### ShapeNet
For the ScanNet and S3DIS datasets, we mainly follow EFEM to segment the chair category, so we can resume their 
SDF data from [link](https://drive.google.com/drive/folders/1qE0Nukw5FcWUqnOR1RUL6gmMgPeFtG73?usp=sharing).  

For our synthetic datasets, the training data of the object-centric network from multiple classes, we firstly download the 
watertight mesh from [link](https://s3.eu-central-1.amazonaws.com/avg-projects/occupancy_networks/data/watertight.zip), 
then follow EFEM to install and use [GAPS](https://github.com/tomfunkhouser/gaps) to create training data of SDF 
on the chosen categories. The watertight mesh folder should be manually reorganized like this:
```shell script
ONet_data
└── 02691156
|   └── 1a04e3eab45ca15dd86060f189eb133.off
|   └── 1a6ad7a24bb89733f412783097373bdc.off
|   └── ...
└── 02828884
...
```

After compiling GAPS and downloading watertight Shapenet data, we can run:
```shell script
python cal_gaps.py
```
By running the above command, you will begin to create SDF training data for the chosen classes. 
**The well-prepared multi-class data can also be directly downloaded [here]()**.

[comment]: <> (*Note that the **chair** data is provided in both prepared data of ours and EFEM, but they are different because EFEM does not release the full preparation code.)

### ScanNet
We exactly follow [Mask3D](https://github.com/JonasSchult/Mask3D) to preprocess the ScanNet dataset. Download the ScanNet
dataset from [here](http://kaldir.vc.in.tum.de/scannet_benchmark/documentation). 
You need to sign the terms of use. Uncompress the folder and move it to  `data/scannet/raw`. Follow Mask3D, we also built 
superpoints by applying Felzenszwalb and Huttenlocher's Graph-Based Image Segmentation algorithm to the test scenes using the default parameters. 

Please come into `ScanNet/Segmentor`, and follow the [official repro](https://github.com/ScanNet/ScanNet/tree/master/Segmentator)
to build by running `make` (or create makefiles for your system using `cmake`). This will create a segmentator binary file. 
Finally, go outside the `ScanNet` to run the segmentator:
```shell script
./run_segmentator.sh your_scannet_tranval_path ## e.g./home/zihui/SSD/ScanNetv2/scans
./run_segmentator.sh your_scannet_test_path ## e.g./home/zihui/SSD/ScanNetv2/scans_test
```
Having the superpoints file, we can run the preprocessing code:
```shell script
python preprocessing/scannet_preprocessing.py
```

### S3DIS
S3DIS dataset can be found [here](
https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1). 
Download the files named "Stanford3dDataset_v1.2_Aligned_Version.zip". Uncompress the folder and move it to 
 `data/s3dis_align/raw`. There is an error in `line 180389` of file `Area_5/hallway_6/Annotations/ceiling_1.txt` 
which needs to be fixed manually and modify the `copy_Room_1.txt` in `Area_6/copyRoom_1` to `copyRoom_1.txt`
.Then run the below commands to begin preprocessing:
```shell script
python preprocessing/s3dis_preprocessong.py
python prepare_superpoints/initialSP_prepare_s3dis_SPG.py
```

### Synthetic Scenes
Download our data from [link]() and put it under the `data/sys_scene/processed`, then run the below command after compiling SPG:
```shell script
python prepare_superpoints/initialSP_prepare_sys_SPG.py
```

### Data Structure
After previous downloading and preprocessing, the data structure should be:
```shell script
data
└── scannet
|   └── raw
|   └── processed
└── s3dis_align
|   └── raw
|   └── processed
|   └── SPG_0.05
└── sys_scene
|   └── processed
|   └── SPG_0.01
└── chairs
|   └── 03001627
|   └── 03001627_dep
└── GAPS_SDF
|   └── 02691156
|   └── 02828884
|   └── ...
└── shapenet_splits
```

## 2. Object-centric Network training
We have two versions of the object-centric network, the first one is for chair segmentation on ScanNet and S3DIS.
Running the command to construct augmentation data for training the Object-centric Network:
```shell script
# Prepare point clouds for more categories as augmentation data in ./data/other_cls_data/
python create_aug_data.py
```
The first chair SDF is trained as below:
```shell script
# Train the rotation estimation part, this will produce a ckpt in ./objnet/pos/
CUDA_VISIBLE_DEVICES=0 python train_vae_chair.py --stage="pos"

# Train the SDF in VAE version, this will produce a ckpt in ./objnet/sdf/
CUDA_VISIBLE_DEVICES=0 python train_vae_chair.py --stage="vae"

# diffusion model (optional): 
# And our latent diffusion model is based on the VAE feature space, so we need to train the VAE at first.
# ddpm
CUDA_VISIBLE_DEVICES=0 python train_ddpm_chair.py
# or rectflow 
CUDA_VISIBLE_DEVICES=0 python train_rectflow_chair.py
```
The second one is for multiple category segmentation on our synthetic scenes and trained as below:
```shell script
# Train the rotation estimation part, this will produce a ckpt in ./multiclass_objnet/pos/
CUDA_VISIBLE_DEVICES=0 python train_vae_multiclass.py --stage="pos"

# Train the SDF in VAE version, this will produce a ckpt in ./multiclass_objnet/sdf/
CUDA_VISIBLE_DEVICES=0 python train_vae_multiclass.py --stage="vae"

# diffusion model (optional): 
CUDA_VISIBLE_DEVICES=0 python train_ddpm_multiclass.py
```
## 3. Object Segmentation Network training
### ScanNet
The well-trained object-centric model for **chair** is saved in ```./objnet/sdf/``` or ```./objnet/diff/``` by default.
We can train a segmentation model on ScanNet by simply running:
```shell script
# Train the segnet by VAE SDF
CUDA_VISIBLE_DEVICES=0 python train_seg_scannet.py

# Train the segnet by ddpm
CUDA_VISIBLE_DEVICES=0 python train_ddpm_scannet.py
# or train it by rectflow
CUDA_VISIBLE_DEVICES=0 python train_rectflowseg_scannet.py
```

### S3DIS
In our main experiments, we conduct a cross-dataset validation that uses the well-trained segmentation model from ScaNet 
to evaluate on S3DIS. For example, to evaluate on S3DIS Area5:
```shell script
# ScanNet to S3DIS eval
CUDA_VISIBLE_DEVICES=0 python train_seg_s3dis.py --use_sp=False --cross_test=True --cross_test_ckpt=your_ckpt # e.g.'ckpt_segnet/scannet_VAE_chair/checkpoint_450.tar'
```

**Optional:** Train Segmentation models on S3DIS.
The training results on S3DIS are not performed in our paper, but we can also do it by: 
```shell script
# Train the segnet by VAE SDF
CUDA_VISIBLE_DEVICES=0 python train_seg_s3dis.py

# Train the segnet by Diffusion SDF
CUDA_VISIBLE_DEVICES=0 python train_ddpmseg_s3dis.py
```

### Synthetic Dataset
The well-trained object-centric models on six categories are saved in ```./objnet/multiclass/sdf/``` or ```./objnet/multiclass/diff/``` by default.
We can train a segmentation model on a Synthetic dataset by simply running:
```shell script
# Train the segnet by VAE SDF
CUDA_VISIBLE_DEVICES=0 python train_seg_sys.py

# Train the segnet by Diffusion SDF
CUDA_VISIBLE_DEVICES=0 python train_ddpmseg_sys.py
```

## 4. Model checkpoints
We also provide the well-trained checkpoints for ScanNet and the synthetic dataset in this [link](), note that the checkpoints for cross-dataset evaluation on S3DIS are also trained on ScanNet.

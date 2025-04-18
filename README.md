[![arXiv](https://img.shields.io/badge/arXiv-2504.11754-b31b1b.svg)](https://arxiv.org/abs/2504.11754)
![code visitors](https://visitor-badge.glitch.me/badge?page_id=vLAR-group/OGC)
[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
[![Twitter Follow](https://img.shields.io/twitter/follow/vLAR_Group?style=social)](https://twitter.com/vLAR_Group)

## GrabS: Generative Embodied Agent for 3D Object Segmentation without Scene Supervision (ICLR 2025 Spotlight)
[Zihui Zhang](https://scholar.google.com.hk/citations?hl=en&user=jiwazT8AAAAJ&view_op=list_works&sortby=pubdate), [Yafei Yang](https://yangyafei1998.github.io/), [Hongtao Wen](https://hatimwen.github.io/), [Bo Yang](https://yang7879.github.io/)

### Overview

We propose an unsupervised framework to separate learning objectness and search objects in 3D scenes.

<img src="figs/framewrok.png" alt="drawing" width=800/>

Our method enables an Agent to search objects in 3D scenes with the aid of Reinforcement Learning:

<img src="figs/traj.png" alt="drawing" width=800/>

## 1. Environment

### Installing dependencies
```shell script
### CUDA 11.3  GCC 9.4
conda env create -f env.yml
source activate GrabS

pip3 install 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' --no-deps

sudo apt-get install libopenblas-dev
git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
cd MinkowskiEngine
git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
python setup.py install --force_cuda --blas=openblas

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

CONDAENV=YOUR_CONDA_ENVIRONMENT_LOCATION ## e.g. /home/zihui/anaconda3/envs/GrabS
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

## 2. Data Preparation
### ShapeNet
We conduct **chair** segmentation on ScanNet and S3DIS datasets. To train an object-centric network for chair, we resume 
SDF data [link](https://drive.google.com/drive/folders/1qE0Nukw5FcWUqnOR1RUL6gmMgPeFtG73?usp=sharing) from EFEM.  

In addition, we create a synthetic dataset and segment it into multiple categories. To collect the training data of the object-centric network for multiple classes, we first download the 
watertight mesh from [link](https://s3.eu-central-1.amazonaws.com/avg-projects/occupancy_networks/data/watertight.zip), 
then follow EFEM to install and use [GAPS](https://github.com/tomfunkhouser/gaps) to compute Ground Truth SDF. The watertight mesh folder should be manually reorganized like this:
```shell script
ONet_data
└── 02691156
|   └── 1a04e3eab45ca15dd86060f189eb133.off
|   └── 1a6ad7a24bb89733f412783097373bdc.off
|   └── ...
└── 02828884
...
```

After compiling GAPS and downloading watertight Shapenet data, we can run the following command to compute Ground Truth SDF:
```shell script
python cal_gaps.py
```
**The well-prepared multi-class data can also be directly downloaded [here]()**.


### ScanNet
We exactly follow [Mask3D](https://github.com/JonasSchult/Mask3D) to preprocess the ScanNet dataset. Download the ScanNet dataset from [here](http://kaldir.vc.in.tum.de/scannet_benchmark/documentation). 
Uncompress the folder and move it to  `data/scannet/raw`. Follow Mask3D, we also built superpoints by applying Felzenszwalb and Huttenlocher's Graph-Based Image Segmentation algorithm to the test scenes using the default parameters. 
Please download the ScanNet tool [link](https://github.com/ScanNet/ScanNet) and come into `ScanNet/Segmentor` to build by running `make` (or create makefiles for your system using `cmake`). This will create a segmentator binary file. 
Finally, go outside the `ScanNet` to run the segmentator:
```shell script
./run_segmentator.sh your_scannet_tranval_path ## e.g ./data/scannet/raw/scans
./run_segmentator.sh your_scannet_test_path ## e.g ./data/scannet/raw/scans_test
```
Having the superpoints file, we can run the preprocessing code:
```shell script
python preprocessing/scannet_preprocessing.py
```

### S3DIS
S3DIS dataset can be found [here](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1). 
Download the files named "Stanford3dDataset_v1.2_Aligned_Version.zip". Uncompress the folder and move it to `data/s3dis_align/raw`. There is an error in `line 180389` of file `Area_5/hallway_6/Annotations/ceiling_1.txt` 
which needs to be fixed manually and modify the `copy_Room_1.txt` in `Area_6/copyRoom_1` to `copyRoom_1.txt`. Then run the below commands to begin preprocessing:
```shell script
python preprocessing/s3dis_preprocessong.py
python prepare_superpoints/initialSP_prepare_s3dis_SPG.py
```

### Synthetic Scenes
Download our data from [Google Drive](https://drive.google.com/file/d/1c422lsYV7c0x0vnPDQnwAbr0J_oImSBF/view?usp=sharing) and put it under the `data/sys_scene/processed`, then run the below command:
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

## 3. Object-centric Network training
We have two versions of the object-centric network in the paper. The first one is for chair segmentation on ScanNet and S3DIS.
To train it, please run the command to construct augmentation data:
```shell script
# Prepare point clouds for more categories as augmentation data in ./data/other_cls_data/
python create_aug_data.py
```
The chair SDF is trained as follows:
```shell script
# Train the rotation estimation part, this will produce a ckpt in ./objnet/chair/pos/
CUDA_VISIBLE_DEVICES=0 python train_vae_chair.py --stage="pos"

# Train the SDF in VAE version, this will produce a ckpt in ./objnet/chair/vae/
CUDA_VISIBLE_DEVICES=0 python train_vae_chair.py --stage="vae"

# diffusion version (optional): 
# And our latent diffusion model is based on the VAE feature space, so we need to train the VAE at first.
# ddpm
CUDA_VISIBLE_DEVICES=0 python train_ddpm_chair.py
# or rectflow 
CUDA_VISIBLE_DEVICES=0 python train_rectflow_chair.py
```
The second object-centric network is for multiple category segmentation on our synthetic scenes and trained as follows:
```shell script
# Train the rotation estimation part, this will produce a ckpt in ./objnet/multi-cate/pos/
CUDA_VISIBLE_DEVICES=0 python train_vae_multiclass.py --stage="pos"

# Train the SDF in VAE version, this will produce a ckpt in ./objnet/multi-cate/vae/
CUDA_VISIBLE_DEVICES=0 python train_vae_multiclass.py --stage="vae"

# diffusion model (optional): 
CUDA_VISIBLE_DEVICES=0 python train_ddpm_multiclass.py
```
## 4. Object Segmentation Network training
### ScanNet
The well-trained object-centric model for **chair** is saved in ```./objnet/chair/vae/``` or ```./objnet/chair/ddpm``` or ``./objnet/chair/rectflow``` by default.
The segmentation model on ScanNet can be trained by:
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
The well-trained object-centric models on multiple categories are saved in ```./objnet/multi-cate/vae/``` or ```./objnet/multi-cate/diff/``` by default.
We can train a segmentation model on a Synthetic dataset by simply running:
```shell script
# Train the segnet by VAE SDF
CUDA_VISIBLE_DEVICES=0 python train_seg_sys.py

# Train the segnet by Diffusion SDF
CUDA_VISIBLE_DEVICES=0 python train_ddpmseg_sys.py
```

## 5. Model checkpoints
We also provide well-trained checkpoints for ScanNet and the synthetic dataset in [Google Drive](https://drive.google.com/file/d/1WoBWTSOvgg4SP_1363y_DjHVrIkBj-Z8/view?usp=sharing). Note that the checkpoints for cross-dataset evaluation on S3DIS are also trained on ScanNet.

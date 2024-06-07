# IR2QSM: Quantitative susceptibility mapping using deep neural networks with iterative Reverse Concatenations and Recurrent Modules

- This repository is a novel deep neural network for quantitative susceptibility mapping (i.e., IR2QSM), enabling QSM reconstrcution from local field.
- This code was built and tested on ubuntu 19.10 with Nvidia Tesla A6000.

## Content

- Overview
  1. Overall Framework
  2. Representative Results
- Manual
  1. Requirements
     - Codes Description
  2. Quick Start(on data)
  3. Reconstruction on your own data
  4. Train new IR2QSM networks

## Overview

### 1.Overall Framework

![LoopNet总结构图5](https://github.com/YangGaoUQ/IR2QSM/assets/58645866/6d8899f4-982a-4bac-b0bc-1063e9899838)


Fig. 1. Overview of the proposed IR2QSM trained from the proposed (a) IR2U-net. (b) depicts the detailed and unrolled view of IR2U-net, which is implemented by iterating a specially tailored U-net for T times. In addition to the conventional (c) Encoder and (d) Decoder blocks, three Reverse Concatenations (the dashed line at the top of (a)) and one (e) middle Recurrent Module module are also introduced in the U-net to enhance the efficiency of the latent feature utilization.

### 2.Representative Results

![image](https://github.com/YangGaoUQ/IR2QSM/assets/58645866/409c8039-6b43-4f2c-92e3-0dca6ba7385c)

Fig. 2. Comparison of IR2QSM with various QSM methods on a COSMOS-based simulated brain at 3T on two orthogonal planes. Error maps from the zoomed-in regions as highlighted by the red boxes are amplified. 

## Manual

### Requirements

```
- Python 3.10 or later
- NVDIA GPU (CUDA 10.0)
- Anaconda Navigator (4.6.11) for Pytorch Installation
- Pytorch 1.13 or later
```

#### Codes Description

```
Two main demo codes showing how to use or train QSM reconstruction:
Evaluate codes:

- test.py ---- For IR2QSM reconstruction
- IR2Unet.py and IR2UnetBlocks.py ---- implementation of IR2Unet

Train codes:
- TrainingDataLoad.py ---- data loader during training
- TrainIR2Unet.py  ---- network training
- IR2Unet.py and IR2UnetBlocks.py ---- implementation of IR2Unet
- loss.py  ---- network loss of training
```

### Quick Start(on data)

1. Clone this repository.

```
git clone https://github.com/YangGaoUQ/IR2QSM
```

2. Install prerequisites (on linux system).

	(1). Installl anaconda (https://docs.anaconda.com/anaconda/install/linux/);

	(2). open a terminal and create your conda environment to install Pytorch and supporting packages (scipy); the following is an example

```
conda create --name Pytorch
conda activate Pytorch
conda install pytorch cudatoolkit=10.2 -c pytorch
conda install scipy
```

### Reconstruction on your own data

1. Change the address of the trained model on line 86 of test.py. (We provide the reference model model_IR2Unet.pth)
2. Use the input image address of the local field and the target output image address in lines 90 and 91 of test.py, respectively. (We provided the reference local field image lfs1.nii)
3. Go to folder 'Evaluate' and run the evaluation codes:

```
python test.py
```

### Train new IR2QSM networks

1. Prepare and preprocess your data.

2. Go to folder 'Train' and run the training codes:

```
python TrainIR2UNet.py
```


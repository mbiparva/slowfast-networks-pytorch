# SlowFast Networks for Video Recognition in PyTorch
This is a PyTorch implementation of the "SlowFast Networks for Video Recognition" paper by Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, Kaiming He published in ICCV 2019. The official code has not been released yet. This implementation is motivated by the code found [here](https://github.com/r1ch88/SlowFastNetworks).

## Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
4. [Datasets](#datasets)
5. [Preparation](#preparation)
6. [Training and Testing Procedures](#training-and-testing-procedures)
7. [Experimental Results](#experimental-results)

## Introduction
Action recognition is one of the core tasks in video understanding and it has similar importance to image classification in the static vision domain. There are two common approaches in deep learning that started far apart at the beginning and recently have shown converging to somewhere in between. The first approach is using 3D convolutional layers that process the input spatiotemporal tensor while the second approach is human-brain-inspired and benefits from a Siamese network architecture. There are two parallel pathway of information processing: one takes RGB frames while the other takes optical flow frames. The "Spatiotemporal Multiplier Networks for Video Action Recognition" paper is an attempt to show how cross-stream lateral connections in a ResNet network architecture could be realized.

## Installation

1. Clone the spatiotemporal-multiplier-networks-pytorch repository

```shell
# Clone the repository
git clone https://github.com/mbiparva/spatiotemporal-multiplier-networks-pytorch.git
```

2. Go into the tools directory

```shell
cd tools
```

3. Run the training or testing script
```shell
# to train
python train.py
# to test
python test.py
```

## Pre-trained Base Networks
Please download the pre-trained base networks provided by the official repository [here](https://github.com/feichtenhofer/st-resnet#models-st-mulnet). The current implementatio uses ResNet-50, so make sure you choose the network snapshot that matches best your dataset (UCF-101), network architecture (ResNet-50), and the dataset split number correctly.
You need to copy the downloaded pre-trained networks in experiment/base_pretrained_nets/ directory to be found by the network module.

## Datasets
You can download the RGB and Optical Flow frames for both UCF-101 and HMDB-51 at the official repository [here](https://github.com/feichtenhofer/st-resnet#models-st-mulnet). You just need to extract the zip files in the dataset directory such that it respect the following directory hierarchy so then the provided dataloader can easily find directories of different categories.

### Directory Hierarchy
Please make sure the downloaded dataset folders and files sit according to the following structure:

```
dataset
|    | UCF101
|    |    | images
│    │    │    | ApplyEyeMakeup  
│    │    │    | ApplyLipstick  
│    │    │    | ...  
|    |    | flows
│    │    │    | ApplyEyeMakeup  
│    │    │    | ApplyLipstick  
│    │    │    | ...  
|    |    | annotations
|    |    |    | annot01.json
|    |    |    | annot02.json
|    |    |    | annot03.json
|    |    |    | ucf101_splits
|    |    |    |    | trainlist01
|    |    |    |    | trainlist02
|    |    |    |    | ....
```
### JSON Annotation Generation
You need to create the annotations of each training and test splits using the script provided in the lib/utils/json_ucf.py. They need to be placed in the annotation folder as described above.

## Preparation
This implementation is tested on the following packages:
* Python 3.7
* PyTorch 1.0 
* CUDA 9.0
* EasyDict

## Training and Testing Procedures
You can train or test the network by using the "train.py" or "test.pt" as follows.

### Training Script
You can use the tools/train.py to start training the network. If you use --help you will see the list of optional sys arguments that could be passed such as "--use-gpu" and "--gpu-id". You can also have a custom cfg file loaded to customize the reference one if you would not like to change the reference one. Additionally, you can set them one by one once you call "--set".

### Test Script
You can use the tools/test.py to start testing the network by loading a custom network snapshot. You have to pass "--pre-trained-id" and "--pre-trained-epoch" to specify the network id and the epoch the snapshot was taken at.

### Configuration File
All of the configuration hyperparameters are set in the lib/utils/config.py. If you want to change them permanently, simply edit the file with the settings you would like to. Otherwise, use the approaches mentioned above to temporary change them.

## Experimental Results
This section will be updated with preliminary results soon.

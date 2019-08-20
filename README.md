# SlowFast Networks for Video Recognition in PyTorch
This is a PyTorch implementation of the "SlowFast Networks for Video Recognition" paper by Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, Kaiming He published in ICCV 2019. The official code has not been released yet. This implementation is motivated by the code found [here](https://github.com/r1ch88/SlowFastNetworks).

## Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Pre-trained Base Networks](#pre-trained-base-networks)
4. [Datasets](#datasets)
5. [Preparation](#preparation)
6. [Training and Testing Procedures](#training-and-testing-procedures)
7. [Experimental Results](#experimental-results)

## Introduction
Action recognition is one of the core tasks in video understanding and it has similar importance to image classification in the static vision domain. There are two common approaches in deep learning that started far apart at the beginning and recently have shown converging to somewhere in between. The first approach is using 3D convolutional layers that process the input spatiotemporal tensor while the second approach is human-brain-inspired and benefits from a Siamese network architecture. Recently, [Christoph](https://arxiv.org/abs/1812.03982) has proposed to extend the two-stream networks with the idea of having expert networks on each pathway: (1) the slow pathway has high parametric capacity and process the RGB information in slower speed of processing while (2) the fast pathway benefits from the fast pathway with a wider temporal receptive field but lower parametric capacity. This has shown promising improvement over the SOTA in action recognition and detection tasks on Kinetics and AVA datasets respectively.

## Installation

1. Clone the slowfast-networks-pytorch repository

```shell
# Clone the repository
git clone https://github.com/mbiparva/slowfast-networks-pytorch.git
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
The pre-trained networks have not officially released yet. The original work trains the proposed network on a GPU cluster of 128 GPUs (i.e. 8 * 8). We provide this implementation as a proof-of-concept model that improves over the previous implementations of this work.

## Datasets
Simply download the UCF101 from the original publisher at [here](https://www.crcv.ucf.edu/data/UCF101.php). You need to extract the zip files in the dataset directory such that it respect the following directory hierarchy so then the provided dataloader can easily find directories of different categories. Please run the jupyter notebook located in the lib/utils accordingly to split the original dataset based on the split number into the training and validation sets.

### Directory Hierarchy
Please make sure the downloaded dataset folders and files sit according to the following structure:

```
dataset
|    | UCF101
|    |    | training
│    │    │    | ApplyEyeMakeup  
│    │    │    | ApplyLipstick  
│    │    │    | ...  
|    |    | validation
│    │    │    | ApplyEyeMakeup  
│    │    │    | ApplyLipstick  
│    │    │    | ...  
```
## Preparation
This implementation is tested on the following packages:
* Python 3.7
* PyTorch 1.2
* CUDA 10.1
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
Currently, the implementation is trained and tested on UCF101 with the following results. We are going to update gradually as we search for hyperparameters that improve the prediction results. The results are the 0-1 classification accuracy rate.

| Net           | Training      | Test  |
| ------------- |:-------------:| -----:|
| Aug 2019      | 92%           | 42%   |

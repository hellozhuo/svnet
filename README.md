# SVNet: Where SO(3) Equivariance Meets Binarization on Point Cloud Representation

This repository contains the PyTorch implementation for 
"SVNet: Where SO(3) Equivariance Meets Binarization on Point Cloud Representation" 
by 
[Zhuo Su](https://zhuogege1943.com/homepage/)\*, 
[Max Welling](https://scholar.google.com/citations?user=8200InoAAAAJ&hl=en), 
[Matti Pietikäinen](https://en.wikipedia.org/wiki/Matti_Pietik%C3%A4inen_(academic)) and 
[Li Liu](http://lilyliliu.com/)\*\* 
(\[[arXiv](https://arxiv.org/abs/2209.05924)\]

The writing style of this code is based on [Pixel Difference Convolution](https://github.com/zhuoinoulu/pidinet).

If you find something useful from our work, please consider citing [our paper](svnet.bib). 

## Introduction

Efficiency and robustness are increasingly needed for applications on 3D point clouds, with the ubiquitous use of edge devices in scenarios like autonomous driving and robotics, which often demand real-time and reliable responses. The paper tackles the challenge by designing a general framework to construct 3D learning architectures with SO(3) equivariance and network binarization. However, a naive combination of equivariant networks and binarization either causes sub-optimal computational efficiency or geometric ambiguity. We propose to locate both scalar and vector features in our networks to avoid both cases. Precisely, the presence of scalar features makes the major part of the network binarizable, while vector features serve to retain rich structural information and ensure SO(3) equivariance. The proposed approach can be applied to general backbones like PointNet and DGCNN. Meanwhile, experiments on ModelNet40, ShapeNet, and the real-world dataset ScanObjectNN, demonstrated that the method achieves a great trade-off between efficiency, rotation robustness, and accuracy.

<div align=center>
<img src="https://user-images.githubusercontent.com/87975270/208468511-a7ca236d-d756-48f9-ba7d-b8fa694d0a89.png"><br>
</div>


## Running environment

Training: Pytorch 1.9 with cuda 10.1 and cudnn 7.5, python 3.6 in an Ubuntu 18.04 system <br>

*Ealier versions may also work~ :)*

## Dataset

To download ModelNet40 and ShapeNet:

```bash
  bash download_datasets.sh
```
The following script will create a foler `data` and download the two datasets in it.

To download ScanObjectNN, please visit [the official website](https://hkust-vgd.github.io/scanobjectnn/) and make a download request.


## Training (without any rotation)

For each section, we provide scripts for both full-precision and binary version to train SVNet. The full-precision one achieves the state-of-the-art accuracy, while the binary version gives a better accuracy-efficiency balance. **For both versions, we are able the train the model without any rotatoin, while test it with random rotation.**

### On ModelNet40

SVNet based on PointNet
```bash
python main_cls_pointnet.py --model=svnet --data-dir data --save-dir result/train --rot aligned --rot-test so3

python main_cls_pointnet.py --model=svnet --data-dir data --save-dir result/train --rot aligned --rot-test so3 --binary --wd 0
```

SVNet based on DGCNN
```bash
python main_cls_dgcnn.py --model=svnet --data-dir data --save-dir result/train --rot aligned --rot-test so3

python main_cls_dgcnn.py --model=svnet --data-dir data --save-dir result/train --rot aligned --rot-test so3 --binary --wd 0
```

### On ShapeNet

SVNet based on PointNet
```bash
python main_partseg_pointnet.py --model=svnet --data-dir data --save-dir result/train --rot aligned --rot-test so3

python main_partseg_pointnet.py --model=svnet --data-dir data --save-dir result/train --rot aligned --rot-test so3 --binary --wd 0
```

SVNet based on DGCNN
```bash
python main_partseg_dgcnn.py --model=svnet --data-dir data --save-dir result/train --rot aligned --rot-test so3

python main_partseg_dgcnn.py --model=svnet --data-dir data --save-dir result/train --rot aligned --rot-test so3 --binary --wd 0
```

### On ScanObjectNN

SVNet based on PointNet
```bash
python main_cls_dgcnn.py --dataset scanobjectnn --model=svnet --data-dir /data/scanobjectnn --save-dir result/train --rot aligned --rot-test so3

python main_cls_dgcnn.py --dataset scanobjectnn --model=svnet --data-dir /data/scanobjectnn --save-dir result/train --rot aligned --rot-test so3 --binary --wd 0
```

## Evaluation (with random rotation on 3D space)

Based on the above script, simply add --test and --rot-test to evaluate the model. For example:
```bash
# To evaluate the DGCNN based binary model on ModelNet40 (assuming the model is saved on checkpoints/sv_dgcnn_binary_modelnet40.pth)
python main_cls_dgcnn.py --model=svnet --data-dir data --save-dir result/test --rot-test so3 --test checkpoints/sv_dgcnn_binary_modelnet40.pth --binary
```

Please see [scripts.sh](scripts.sh) for more details.

## Network complexity

For example, if you want to check model size and FLOPs/ADDs/BOPs of DGCNN based binary SVNet:
```bash
python params_macs/sv_dgcnn.py
```

Please see [scripts.sh](scripts.sh) for more details.

The performance of some of the models are listed below (click the items to download the checkpoints and training logs). It should be noted that `*i*/so(3)` got the similar results:

| Dataset | Backbone | Version | Knowledge distillation | ACC (*z*/so(3)) | Training logs | Checkpoint |
|---------|----------|---------|------------------------|-----------------|---------------|------------|
| ModelNet40 | PointNet | Full-precision |  | 86.3 | [log](logs/sv_pointnet_fp_modelnet40.txt) | [link](checkpoints/sv_pointnet_fp_modelnet40.pth) |
| ModelNet40 | PointNet | Binary |  | 76.3 | [log](logs/sv_pointnet_binary_modelnet40.txt) | [link](checkpoints/sv_pointnet_binary_modelnet40.pth) |
| ModelNet40 | DGCNN | Full-precision |  | 90.3 | [log](logs/sv_dgcnn_fp_modelnet40.txt) | [link](checkpoints/sv_dgcnn_fp_modelnet40.pth) |
| ModelNet40 | DGCNN | Binary |  | 83.8 | [log](logs/sv_dgcnn_binary_modelnet40.txt) | [link](checkpoints/sv_dgcnn_binary_modelnet40.pth) |
| ModelNet40 | DGCNN | Binary | ✔ | 86.8 | [log](logs/sv_dgcnn_binary_kd_modelnet40.txt) | [link](checkpoints/sv_dgcnn_binary_kd_modelnet40.pth) |
|---------|----------|---------|------------------------|-----------------|---------------|------------|
| ShapeNet | PointNet | Full-precision |  | 78.2 | | [link](checkpoints/sv_pointnet_fp_shapenet.pth) |
| ShapeNet | PointNet | Binary |  | 67.3 | | [link](checkpoints/sv_pointnet_binary_shapenet.pth) |
| ShapeNet | DGCNN | Full-precision |  | 81.4 | [log](logs/sv_dgcnn_fp_shapenet.txt) | [link](checkpoints/sv_dgcnn_fp_shapenet.pth) |
| ShapeNet | DGCNN | Binary |  | 68.4 | [log](logs/sv_dgcnn_binary_shapenet.txt) | [link](checkpoints/sv_dgcnn_binary_shapenet.pth) |
| ShapeNet | DGCNN | Binary | ✔ | 71.5 | [log](logs/sv_dgcnn_binary_kd_shapenet.txt) | [link](checkpoints/sv_dgcnn_binary_kd_shapenet.pth) |
|---------|----------|---------|------------------------|-----------------|---------------|------------|
| ScanObjectNN | DGCNN | Full-precision |  | 76.2 | [log](logs/sv_dgcnn_fp_scanobjectnn.txt) | [link](checkpoints/sv_dgcnn_fp_scanobjectnn.pth) |
| ScanObjectNN | DGCNN | Binary |  | 52.9 | [log](logs/sv_dgcnn_binary_scanobjectnn.txt) | [link](checkpoints/sv_dgcnn_binary_scanobjectnn.pth) |
| ScanObjectNN | DGCNN | Binary | ✔ | 60.9 | [log](logs/sv_dgcnn_binary_kd_scanobjectnn.txt) | [link](checkpoints/sv_dgcnn_binary_kd_scanobjectnn.pth) |


## Acknowledgement

We greatly thank the following repos:

- [Vector Neurons](https://github.com/FlyingGiraffe/vnn-pc)
- [DGCNN pytorch](https://github.com/antao97/dgcnn.pytorch)
- [PointNet pytorch](https://github.com/fxia22/pointnet.pytorch)


## License
MIT License


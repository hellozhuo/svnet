#!/bin/bash

mkdir data
cd data

# Download HDF5 for ModelNet40
wget https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip --no-check-certificate
unzip modelnet40_ply_hdf5_2048.zip
rm modelnet40_ply_hdf5_2048.zip

# Download HDF5 for ShapeNet Part segmentation (around 346MB)
wget https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip --no-check-certificate
unzip shapenet_part_seg_hdf5_data.zip
rm shapenet_part_seg_hdf5_data.zip
mv hdf5_data shapenet_part_seg_hdf5_data

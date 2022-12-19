### Train (we train without any rotation, then test with random rotation in 3D space)

## On ModelNet40
## Based on PointNet, full-precision
python main_cls_pointnet.py --model=svnet --data-dir data --save-dir result/train --rot aligned --rot-test so3

## Based on PointNet, binary
python main_cls_pointnet.py --model=svnet --data-dir data --save-dir result/train --rot aligned --rot-test so3 --binary --wd 0

## Based on DGCNN, full-precision
python main_cls_dgcnn.py --model=svnet --data-dir data --save-dir result/train --rot aligned --rot-test so3

## Based on DGCNN, binary
python main_cls_dgcnn.py --model=svnet --data-dir data --save-dir result/train --rot aligned --rot-test so3 --binary --wd 0


## On ShapeNet
## Based on PointNet, full-precision
python main_partseg_pointnet.py --model=svnet --data-dir data --save-dir result/train --rot aligned --rot-test so3

## Based on PointNet, binary
python main_partseg_pointnet.py --model=svnet --data-dir data --save-dir result/train --rot aligned --rot-test so3 --binary --wd 0

## Based on DGCNN, full-precision
python main_partseg_dgcnn.py --model=svnet --data-dir data --save-dir result/train --rot aligned --rot-test so3

## Based on DGCNN, binary
python main_partseg_dgcnn.py --model=svnet --data-dir data --save-dir result/train --rot aligned --rot-test so3 --binary --wd 0


## On ScanObjectNN
## Based on DGCNN, full-precision
python main_cls_dgcnn.py --dataset scanobjectnn --model=svnet --data-dir /data/scanobjectnn --save-dir result/train --rot aligned --rot-test so3

## Based on DGCNN, binary
python main_cls_dgcnn.py --dataset scanobjectnn --model=svnet --data-dir /data/scanobjectnn --save-dir result/train --rot aligned --rot-test so3 --binary --wd 0



### Evaluation (with random rotation in 3D space)

## On ModelNet40
## Based on PointNet, full-precision
python main_cls_pointnet.py --model=svnet --batch-size 16 --data-dir data --save-dir result/test --rot-test so3 --test checkpoints/sv_pointnet_fp_modelnet40.pth

## Based on PointNet, binary
python main_cls_pointnet.py --model=svnet --batch-size 16 --data-dir data --save-dir result/test --rot-test so3 --test checkpoints/sv_pointnet_binary_modelnet40.pth  --binary

## Based on DGCNN, full-precision
python main_cls_dgcnn.py --model=svnet --batch-size 16 --data-dir data --save-dir result/test --rot-test so3 --test checkpoints/sv_dgcnn_fp_modelnet40.pth

## Based on DGCNN, binary
python main_cls_dgcnn.py --model=svnet --batch-size 16 --data-dir data --save-dir result/test --rot-test so3 --test checkpoints/sv_dgcnn_binary_modelnet40.pth --binary
# with knowledge distillation
python main_cls_dgcnn.py --model=svnet --batch-size 16 --data-dir data --save-dir result/test --rot-test so3 --test checkpoints/sv_dgcnn_binary_kd_modelnet40.pth --binary


## On ShapeNet
## Based on PointNet, full-precision
python main_partseg_pointnet.py --model=svnet --batch-size 4 --data-dir data --save-dir result/test --rot-test so3 --test checkpoints/sv_pointnet_fp_shapenet.pth

## Based on PointNet, binary
python main_partseg_pointnet.py --model=svnet --batch-size 4 --data-dir data --save-dir result/test --rot-test so3 --test checkpoints/sv_pointnet_binary_shapenet.pth --binary

## Based on DGCNN, full-precision
python main_partseg_dgcnn.py --model=svnet --batch-size 4 --data-dir data --save-dir result/test --rot-test so3 --test checkpoints/sv_dgcnn_fp_shapenet.pth

## Based on DGCNN, binary
python main_partseg_dgcnn.py --model=svnet --batch-size 4 --data-dir data --save-dir result/test --rot-test so3 --test checkpoints/sv_dgcnn_binary_shapenet.pth --binary
# with knowledge distillation
python main_partseg_dgcnn.py --model=svnet --batch-size 4 --data-dir data --save-dir result/test --rot-test so3 --test checkpoints/sv_dgcnn_binary_kd_shapenet.pth --binary


## On ScanObjectNN
## Based on DGCNN, full-precision
python main_cls_dgcnn.py --dataset scanobjectnn --model=svnet --batch-size 16 --data-dir /data2/zhuo/dataset/pointcloud/scanobjectnn --save-dir zhuo_test/test --rot-test so3 --test checkpoints/sv_dgcnn_fp_scanobjectnn.pth

## Based on DGCNN, binary
python main_cls_dgcnn.py --dataset scanobjectnn --model=svnet --batch-size 16 --data-dir /data2/zhuo/dataset/pointcloud/scanobjectnn --save-dir zhuo_test/test --rot-test so3 --test checkpoints/sv_dgcnn_binary_scanobjectnn.pth --binary
# with knowledge distillation
python main_cls_dgcnn.py --dataset scanobjectnn --model=svnet --batch-size 16 --data-dir /data2/zhuo/dataset/pointcloud/scanobjectnn --save-dir zhuo_test/test --rot-test so3 --test checkpoints/sv_dgcnn_binary_kd_scanobjectnn.pth --binary



### Check model size and FLOPs/BOPs

python params_macs/sv_pointnet.py
python params_macs/sv_dgcnn.py

python params_macs/vn_pointnet.py
python params_macs/vn_dgcnn.py

python params_macs/pointnet.py

python params_macs/dgcnn.py

python params_macs/bipointnet.py

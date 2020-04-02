## GADH_Net_EA: A Geometry-Attentional Network for ALS Point Cloud Classification
![GA-Conv](https://github.com/WuzhaoLee/GADH_Net_EA/blob/master/doc/GA-Conv.png)
## Introduction
Airborne Laser Scanning (ALS) point cloud classification is a critical task in remote sensing and
photogrammetry communities. In particular, the characteristics of ALS point clouds are distinctive
in three aspects, (1) numerous geometric instances (e.g. tracts of roofs); (2) drastic scale variations
between different categories (e.g. car v.s. roof); (3) discrepancy distribution along the elevation,
which should be specifically focused on for ALS point cloud classification. In this paper, we propose
a geometry-attentional network consisting of geometry-aware convolution, dense hierarchical architecture
and elevation-attention module to embed the three characteristics effectively, which can be
trained in an end-to-end manner. Evaluated on the ISPRS Vaihingen 3D Semantic Labeling benchmark,
our method achieves the state-of-the-art performance in terms of average F1 score and overall
accuracy (OA). Additionally, without retraining, our model trained on the above Vaihingen 3D dataset
can also achieve a better result on the dataset of 2019 IEEE GRSS Data Fusion Contest 3D point cloud
classification challenge (DFC 3D) than the baseline (i.e. PointSIFT), which verifies the stronger generalization
ability of our model.
In this repository, we release code and data for training and inferencing our geometry-attentional network.
## Installation
A docker container implementation has been provided for easy setup. The container is based on a tensorflow-gpu image; therefore, nvidia-docker (version 2) must be installed on the host machine. See https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0) for installation instructions.

The docker container image can be bulit by running:
```
cd /path/to/GADH_Net_EA
docker build . -t ga_net
```
The following steps assume the container name is ```ga_net```

## Create ISPRS dataset
You can download our processed data：block.pickle, or run the following to generate these data.
```
docker run  -it --rm \
    -v /path/to/GADH_Net_EA/data:/pointnet2/data \
    -v /path/to/GADH_Net_EA/sem_seg:/pointnet2/sem_seg \
    ga_net python /pointnet2/sem_seg/create_ISPRS_mydata.py \
    -i /pointnet2/data/ISPRSdata  -o /pointnet2/data/ISPRSdata/block_pickle
```
## Training the model
To train our geometry-attentional network with deep supervision, run:
```
docker run --runtime=nvidia -it --rm \
    -v /path/to/GADH_Net_EA/data:/pointnet2/data \
    -v /path/to/GADH_Net_EA/sem_seg:/pointnet2/sem_seg \
    -v /path/to/GADH_Net_EA/models:/pointnet2/models \
    -v /path/to/GADH_Net_EA/utils:/pointnet2/utils \
    ga_net python /pointnet2/sem_seg/train_multi_gpu_deep.py \
    --data_dir=data/block_pickle \
    --log_dir=data/model1 \
    --model=GADH_Net_EA --extra-dims 3 4 --gpu_num=2
```
## Running Inference
To classify point clouds, first run the following to generate intermediate classification results ```Eval_can_out/Eval_can.txt```, and then use knnVote.m to perform Knn voting to get point-wise semantic labels of the original point cloud ```myout1/EVAL_CLS.txt```:
```
docker run --runtime=nvidia -it --rm \
    -v /path/to/GADH_Net_EA/data:/pointnet2/data \
    -v /path/to/GADH_Net_EA/sem_seg:/pointnet2/sem_seg \
    -v /path/to/GADH_Net_EA/models:/pointnet2/models \
    -v /path/to/GADH_Net_EA/utils:/pointnet2/utils \
    ga_net python /pointnet2/sem_seg/inference_deep.py \
    --model=GANH_Net_EA --extra-dims 3 4 \
    --model_path=data/model1/mode.ckpt-####.ckpt \
    --input_path=data/Inference \
    --output_path=data/Inference/Eval_can_out
```
## evaluate and generate the confusion matrix
run the following:
```
docker run -it --rm \
    -v /path/to/GADH_Net_EA/data/Inference:/data \
    -v /path/to/GADH_Net_EA/sem_seg:/metrics \
    ga_net -c \
   "python /metrics/evaluate.py -g /data/gt_test -d /data/myout1 | tee /data/myout1/metrics_myout1.txt"
```

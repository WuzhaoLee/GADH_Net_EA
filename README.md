# GADH_Net_EA
 A Geometry-Attentional Network for ALS Point Cloud Classification
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



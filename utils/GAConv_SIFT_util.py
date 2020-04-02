"""
wrappers for pointSIFT module
Author: Jiang Mingyang
Email: jmydurant@sjtu.edu.cn
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/pointSIFT_op'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate
from pointSIFT_op import pointSIFT_select, pointSIFT_select_four

import tf_util
import tensorflow as tf
import numpy as np
import math
import time

def pointSIFT_group(radius, xyz, points, use_xyz=True):
    idx = pointSIFT_select(xyz, radius)
    grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, 8, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(xyz, 2), [1, 1, 8, 1])  # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx)  # (batch_size, npoint, 8, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)  # (batch_size, npoint, 8/32, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return xyz, new_points, idx, grouped_xyz

def pointSIFT_group_with_idx(xyz, idx, points, use_xyz=True):
    grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, 8, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(xyz, 2), [1, 1, 8, 1])  # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx)  # (batch_size, npoint, 8/32, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)  # (batch_size, npoint, 8/32, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz
    return xyz, new_points, idx, grouped_xyz

def pointSIFT_group_four(radius, xyz, points, use_xyz=True):
    idx = pointSIFT_select_four(xyz, radius)
    grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, 32, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(xyz, 2), [1, 1, 32, 1])  # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx)  # (batch_size, npoint, 8/32, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)  # (batch_size, npoint, 8/32, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return xyz, new_points, idx, grouped_xyz

def pointSIFT_group_four_with_idx(xyz, idx, points, use_xyz=True):
    grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, 8/32, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(xyz, 2), [1, 1, 32, 1])  # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx)  # (batch_size, npoint, 8/32, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)  # (batch_size, npoint, 8/32, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz
    return xyz, new_points, idx, grouped_xyz

def pointSIFT_module(xyz, points, radius, out_channel, is_training, bn_decay, scope='point_sift', bn=True, use_xyz=True, use_nchw=False):
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        # Grouping
        new_xyz, new_points, idx, grouped_xyz = pointSIFT_group(radius, xyz, points, use_xyz)

        # Point Feature Embedding
        if use_nchw: new_points = tf.transpose(new_points, [0, 3, 1, 2])
        for i in range(3): # This is 3 times conv
            new_points = tf_util.conv2d(new_points, out_channel, [1, 2],
                                        padding='VALID', stride=[1, 2],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d' % (i), bn_decay=bn_decay,
                                        data_format=data_format)
        # add fc
        new_points = tf_util.conv2d(new_points, out_channel, [1, 1],
                                    padding='VALID', stride=[1, 1],
                                    bn=bn, is_training=is_training,
                                    scope='conv_fc', bn_decay=bn_decay,
                                    data_format=data_format)
        if use_nchw: new_points = tf.transpose(new_points, [0, 2, 3, 1])

        new_points = tf.squeeze(new_points, [2])  # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points, idx

def pointSIFT_res_module(xyz, points, radius, out_channel, is_training, bn_decay, scope='point_sift', bn=True, use_xyz=True, same_dim=False, merge='add'):
    data_format = 'NHWC'
    with tf.variable_scope(scope) as sc:
        # conv1
        _, new_points, idx, _ = pointSIFT_group(radius, xyz, points, use_xyz=use_xyz)

        for i in range(3):
            new_points = tf_util.conv2d(new_points, out_channel, [1, 2],
                                        padding='VALID', stride=[1, 2],
                                        bn=bn, is_training=is_training,
                                        scope='c0_conv%d' % (i), bn_decay=bn_decay,
                                        data_format=data_format)
        new_points = tf.squeeze(new_points, [2])
        # conv2
        _, new_points, idx, _ = pointSIFT_group_with_idx(xyz, idx=idx, points=new_points, use_xyz=use_xyz)

        for i in range(3):
            if i == 2:
                act = None
            else:
                act = tf.nn.relu
            new_points = tf_util.conv2d(new_points, out_channel, [1, 2],
                                        padding='VALID', stride=[1, 2],
                                        bn=bn, is_training=is_training,
                                        scope='c1_conv%d' % (i), bn_decay=bn_decay,
                                        activation_fn=act,
                                        data_format=data_format)
        new_points = tf.squeeze(new_points, [2])
        # residual part..
        if points is not None:
            if same_dim is True:
                points = tf_util.conv1d(points, out_channel, 1, padding='VALID', bn=bn, is_training=is_training, scope='merge_channel_fc', bn_decay=bn_decay)
            if merge == 'add':
                new_points = new_points + points
            elif merge == 'concat':
                new_points = tf.concat([new_points, points], axis=-1)
            else:
                print("ways not found!!!")
        new_points = tf.nn.relu(new_points)
        return xyz, new_points, idx


"""
GA-Conv module
Author: Wuzhao Li
Email: wuzhaoli@whu.edu.cn
"""

def compute_determinant(A):
    return A[..., 0, 0] * (A[..., 1, 1] * A[..., 2, 2] - A[..., 1, 2] * A[..., 2, 1]) \
           - A[..., 0, 1] * (A[..., 1, 0] * A[..., 2, 2] - A[..., 1, 2] * A[..., 2, 0]) \
           + A[..., 0, 2] * (A[..., 1, 0] * A[..., 2, 1] - A[..., 1, 1] * A[..., 2, 0])

def compute_eigenvals(A):
    A_11 = A[:, :, 0, 0]  # (N, P)
    A_12 = A[:, :, 0, 1]
    A_13 = A[:, :, 0, 2]
    A_22 = A[:, :, 1, 1]
    A_23 = A[:, :, 1, 2]
    A_33 = A[:, :, 2, 2]
    I = tf.eye(3)
    p1 = tf.square(A_12) + tf.square(A_13) + tf.square(A_23)  # (N, P)
    q = tf.trace(A) / 3  # (N, P)
    p2 = tf.square(A_11 - q) + tf.square(A_22 - q) + tf.square(A_33 - q) + 2 * p1  # (N, P)
    p = tf.sqrt(p2 / 6) + 1e-8  # (N, P)
    N = tf.shape(A)[0]
    q_4d = tf.reshape(q, (N, -1, 1, 1))  # (N, P, 1, 1)
    p_4d = tf.reshape(p, (N, -1, 1, 1))
    B = (1 / p_4d) * (A - q_4d * I)  # (N, P, 3, 3)
    r = tf.clip_by_value(compute_determinant(B) / 2, -1, 1)  # (N, P)
    phi = tf.acos(r) / 3  # (N, P)
    eig1 = q + 2 * p * tf.cos(phi)  # (N, P)
    eig3 = q + 2 * p * tf.cos(phi + (2 * math.pi / 3))
    eig2 = 3 * q - eig1 - eig3  # 矩阵的迹等于特征值之和。
    return tf.abs(tf.stack([eig1, eig2, eig3], axis=2))  # (N, P, 3)

def get_neigh_geo_feat(new_xyz, grouped_xyz, nsample):
    grouped_xyz_mean = tf.reduce_mean(grouped_xyz, 2)
    grouped_xyz_offset = grouped_xyz - tf.tile(tf.expand_dims(grouped_xyz_mean, 2), [1, 1, nsample, 1])
    grouped_xyz_offset1 = tf.expand_dims(grouped_xyz_offset, -1)
    grouped_xyz_offset2 = tf.expand_dims(grouped_xyz_offset, 3)
    cov_in = tf.multiply(grouped_xyz_offset1, grouped_xyz_offset2)
    cov_sum = tf.reduce_sum(cov_in, 2)
    cov_out = tf.divide(cov_sum, nsample)

    # The general operation is too slow, discard it
    # eigenvalues, eigenvectors = tf.self_adjoint_eig(cov_out)

    eigenvalues = compute_eigenvals(cov_out)

    # Eigenvalues are normalized to sum up to 1
    e_sum = tf.tile(tf.expand_dims(tf.reduce_sum(eigenvalues, 2), 2), [1, 1, 3])
    eigenvalues = tf.div(eigenvalues, e_sum)

    lamba1 = tf.slice(eigenvalues, [0, 0, 0], [-1, -1, 1])
    lamba2 = tf.slice(eigenvalues, [0, 0, 1], [-1, -1, 1])
    lamba3 = tf.slice(eigenvalues, [0, 0, 2], [-1, -1, 1])

    feat_Sum = lamba1 + lamba2 + lamba3
    feat_Omnivariance = tf.pow((lamba1 * lamba2 * lamba3), 1 / 3)
    feat_Eigenentropy = tf.negative(lamba1 * tf.log(lamba1) + lamba2 * tf.log(lamba2) + lamba3 * tf.log(lamba3))
    feat_Anisotropy = (lamba1 - lamba3) / lamba1
    feat_Planarity = (lamba2 - lamba3) / lamba1
    feat_Linearity = (lamba1 - lamba2) / lamba1
    feat_Surface_Variation = lamba3 / (lamba1 + lamba2 + lamba3)
    feat_Sphericity = lamba3 / lamba1

    zmax = tf.reduce_max(grouped_xyz, [2])  # reduce_max x,y,z in neighborhood
    zmax = tf.slice(zmax, [0, 0, 2], [-1, -1, 1])
    zmax_tile = tf.tile(tf.expand_dims(zmax, -1), [1, 1, nsample, 1])

    zmin = tf.reduce_min(grouped_xyz, [2])
    zmin = tf.slice(zmin, [0, 0, 2], [-1, -1, 1])
    zmin_tile = tf.tile(tf.expand_dims(zmin, -1), [1, 1, nsample, 1])
    zmax_zmin = zmax - zmin

    z = tf.slice(grouped_xyz, [0, 0, 0, 2], [-1, -1, -1, 1]) # 16,1024,32,1
    z_mean,z_var=tf.nn.moments(z,axes=[2])  # 16,1024,1
    z_var = tf.tile(tf.expand_dims(z_var, -1), [1, 1, nsample, 1]) # 16*1024*32*1
    zmax_z = zmax_tile - z  # (16,1024,32,1)
    z_zmin = z - zmin_tile  # (16,1024,32,1)


    neigh_geofeat = tf.concat([feat_Sum, feat_Omnivariance, feat_Eigenentropy, feat_Anisotropy, feat_Planarity,
                               feat_Linearity, feat_Surface_Variation, feat_Sphericity, zmax_zmin], axis=2)   # B * N * 9
    neigh_geofeat = tf.tile(tf.expand_dims(neigh_geofeat, 2), [1, 1, nsample,1])

    neigh_geofeat=tf.concat([neigh_geofeat, z_var, zmax_z,z_zmin],axis=-1)  # B *N *32 * 12
    return neigh_geofeat

# def get_hij(new_xyz, grouped_xyz, nsample):
#     grouped_xyz_centroid = tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])
#     xi_xj = grouped_xyz - grouped_xyz_centroid
#     xi=grouped_xyz_centroid
#     xj=grouped_xyz
#     distance=tf.sqrt(tf.reduce_sum(tf.square(xi_xj),axis=-1))
#     distance=tf.expand_dims(distance,-1)
#     hij=tf.concat([distance,xi_xj,xi,xj],axis=-1)  # B *N *32 * 10
#     return hij

def pointnet_GAconv(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope,
                       bn=True, pooling='max', knn=False, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        # Sample and Grouping
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz ,G_p= sample_and_group(npoint, radius, nsample, xyz, points, knn, use_xyz)

        # Point Feature Embedding
        num_channel = new_points.get_shape()[-1].value
        M = [num_channel,num_channel]  # note this
        for i, num_out in enumerate(M):
            G_p = tf_util.conv2d(G_p, num_out, [1, 1],
                                            padding='VALID', stride=[1, 1],
                                            bn=bn, is_training=is_training,
                                            scope='convNei%d' % (i), bn_decay=bn_decay,
                                            data_format=data_format)

        G_p = tf.nn.softmax(G_p)
        new_points = tf.multiply(G_p, new_points)

        if use_nchw: new_points = tf.transpose(new_points, [0, 3, 1, 2])

        # Pooling in Local Regions , note that the dim is consistent.
        if pooling == 'max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
        elif pooling == 'avg':
            new_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
        elif pooling == 'weighted_avg':
            with tf.variable_scope('weighted_avg'):
                dists = tf.norm(grouped_xyz, axis=-1, ord=2, keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists / tf.reduce_sum(exp_dists, axis=2,
                                                    keep_dims=True)  # (batch_size, npoint, nsample, 1)
                new_points *= weights  # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif pooling == 'max_and_avg':
            max_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
            avg_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        # further channel-wise mapping
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1, 1],
                                        padding='VALID', stride=[1, 1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d' % (i), bn_decay=bn_decay,
                                        data_format=data_format)  # 16*1024*32*128

        if use_nchw: new_points = tf.transpose(new_points, [0, 2, 3, 1])
        #new_points=tf.concat([new_points,neigh_geo_feat],axis=-1)

        # [Optional] Further Processing
        if mlp2 is not None:
            if use_nchw: new_points = tf.transpose(new_points, [0, 3, 1, 2])
            for i, num_out_channel in enumerate(mlp2):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1, 1],
                                            padding='VALID', stride=[1, 1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_post_%d' % (i), bn_decay=bn_decay,
                                            data_format=data_format)
            if use_nchw: new_points = tf.transpose(new_points, [0, 2, 3, 1])

        new_points = tf.squeeze(new_points, [2])  # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points, idx
"""
PointNet++ layers
Author: Charles R. Qi
Date: November 2017
"""


def sample_and_group(npoint, radius, nsample, xyz, points, knn=False, use_xyz=True):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''

    new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))  # (batch_size, npoint, 3)  B*N*3
    if knn:
        _, idx = knn_point(nsample, xyz, new_xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
        # idx (batch_size, npoint, nsample)  pts_cnt (batch_size, npoint)
    grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, nsample, 3)

    G_p= get_neigh_geo_feat(new_xyz,grouped_xyz,nsample) # B * N * 32 *12

    # now, grouped_xyz is local coordinate
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])  # translation normalization


    if points is not None:
        grouped_points = group_point(points, idx)  # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)  # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz, G_p


def sample_and_group_all(xyz, points, use_xyz=True):
    '''
    Inputs:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (batch_size, 1, 3) as (0,0,0)
        new_points: (batch_size, 1, ndataset, 3+channel) TF tensor
    Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    '''
    batch_size = xyz.get_shape()[0].value
    nsample = xyz.get_shape()[1].value
    new_xyz = tf.constant(np.tile(np.array([0, 0, 0]).reshape((1, 1, 3)), (batch_size, 1, 1)),
                          dtype=tf.float32)  # (batch_size, 1, 3)
    idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1, 1, nsample)), (batch_size, 1, 1)))
    grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3))  # (batch_size, npoint=1, nsample, 3)
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2)  # (batch_size, 16, 259)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1)  # (batch_size, 1, 16, 259)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz


def pointnet_sa_module_msg(xyz, points, npoint, radius_list, nsample_list, mlp_list, is_training, bn_decay, scope,
                           bn=True, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: list of float32 -- search radius in local region
            nsample: list of int32 -- how many points in each local region
            mlp: list of list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, \sum_k{mlp[k][-1]}) TF tensor
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))
        new_points_list = []
        for i in range(len(radius_list)):
            radius = radius_list[i]
            nsample = nsample_list[i]
            idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = group_point(xyz, idx)
            grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])
            if points is not None:
                grouped_points = group_point(points, idx)
                if use_xyz:
                    grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)
            else:
                grouped_points = grouped_xyz
            if use_nchw: grouped_points = tf.transpose(grouped_points, [0, 3, 1, 2])
            for j, num_out_channel in enumerate(mlp_list[i]):
                grouped_points = tf_util.conv2d(grouped_points, num_out_channel, [1, 1],
                                                padding='VALID', stride=[1, 1], bn=bn, is_training=is_training,
                                                scope='conv%d_%d' % (i, j), bn_decay=bn_decay)
            if use_nchw: grouped_points = tf.transpose(grouped_points, [0, 2, 3, 1])
            new_points = tf.reduce_max(grouped_points, axis=[2])
            new_points_list.append(new_points)
        new_points_concat = tf.concat(new_points_list, axis=-1)
        return new_xyz, new_points_concat


def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope, bn=True):
    ''' PointNet Feature Propogation (FP) Module
        Input:
            xyz1: (batch_size, ndataset1, 3) TF tensor
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
            points1: (batch_size, ndataset1, nchannel1) TF tensor
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0 / dist), axis=2, keep_dims=True)
        norm = tf.tile(norm, [1, 1, 3])
        weight = (1.0 / dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1])  # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1, 1],
                                         padding='VALID', stride=[1, 1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_%d' % (i), bn_decay=bn_decay)
        new_points1 = tf.squeeze(new_points1, [2])  # B,ndataset1,mlp[-1]
        return new_points1

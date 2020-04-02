import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR,'../utils'))

import tensorflow as tf
import tf_util
from GAConv_SIFT_util import pointSIFT_module, pointSIFT_res_module, pointnet_fp_module, pointnet_GAconv

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 5))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    smpws_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl, smpws_pl


def get_model(point_cloud, is_training, num_class, bn_decay=None):
    """ Our GANH-Net with EA, the input is B x N x 5, output B x num_class """
    end_points = {}
    l0_xyz = tf.slice(point_cloud, [0, 0, 0], [-1, -1, 3])

    # Elevation-attention module;
    # 3-layer MLP; use the softmax as normalized layer; You can replace it with sigmoid.
    l0_z = tf.slice(point_cloud, [0, 0, 2], [-1, -1, 1])   # B*N*1
    l0_z_weight = tf_util.conv1d(l0_z, 64, 1, padding='VALID', bn=True, is_training=is_training, scope='l0_z_weight1',bn_decay=bn_decay)
    l0_z_weight = tf_util.conv1d(l0_z_weight, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='l0_z_weight2',bn_decay=bn_decay)
    l0_z_weight = tf_util.conv1d(l0_z_weight, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='l0_z_weight3',bn_decay=bn_decay)
    l0_z_weight = tf.nn.softmax(l0_z_weight)

    l0_points = tf.slice(point_cloud, [0, 0, 3], [-1, -1, 2])
    end_points['l0_xyz'] = l0_xyz

    # PointSIFT embedding
    c0_l0_xyz, c0_l0_points, c0_l0_indices = pointSIFT_res_module(l0_xyz, l0_points, radius=0.1, out_channel=64, is_training=is_training, bn_decay=bn_decay, scope='layer0_c0', merge='concat')

    # GA-Conv for feature encoding
    l1_xyz, l1_points, l1_indices = pointnet_GAconv(c0_l0_xyz, c0_l0_points, npoint=1024, radius=0.1, nsample=32, mlp=[64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')

    # PointSIFT embedding
    c0_l1_xyz, c0_l1_points, c0_l1_indices = pointSIFT_res_module(l1_xyz, l1_points, radius=0.25, out_channel=128, is_training=is_training, bn_decay=bn_decay, scope='layer1_c0')

    # Nested architecture
    l01_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, c0_l1_points, [128], is_training, bn_decay, scope='u01_layer') # B*N*128
    l01_points = tf.concat([l01_points,c0_l0_points],axis=-1)
    l01_points=tf_util.conv1d(l01_points,128, 1, padding='VALID', bn=True, is_training=is_training, scope='l01', bn_decay=bn_decay)

    # GA-Conv for feature encoding
    l2_xyz, l2_points, l2_indices = pointnet_GAconv(c0_l1_xyz, c0_l1_points, npoint=256, radius=0.2, nsample=32, mlp=[128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')

    # PointSIFT embedding
    c0_l2_xyz, c0_l2_points, c0_l2_indices = pointSIFT_res_module(l2_xyz, l2_points, radius=0.5, out_channel=256, is_training=is_training, bn_decay=bn_decay, scope='layer2_c0')
    c1_l2_xyz, c1_l2_points, c1_l2_indices = pointSIFT_res_module(c0_l2_xyz, c0_l2_points, radius=0.5, out_channel=512, is_training=is_training, bn_decay=bn_decay, scope='layer2_c1', same_dim=True)
    l2_cat_points = tf.concat([c0_l2_points, c1_l2_points], axis=-1)
    fc_l2_points = tf_util.conv1d(l2_cat_points, 512, 1, padding='VALID', bn=True, is_training=is_training, scope='conv_2_fc', bn_decay=bn_decay)

    # Nested architecture
    l11_points = pointnet_fp_module(l1_xyz, l2_xyz, None, fc_l2_points, [128], is_training, bn_decay,scope='u11_layer')
    l11_points = tf.concat([l11_points,c0_l1_points],axis=-1)
    l11_points = tf_util.conv1d(l11_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='l11',bn_decay=bn_decay)

    l02_points = pointnet_fp_module(l0_xyz, l1_xyz, None, l11_points, [128], is_training, bn_decay,scope='u02_layer')
    l02_points = tf.concat([l02_points,l01_points,c0_l0_points],axis=-1)
    l02_points = tf_util.conv1d(l02_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='l02',bn_decay=bn_decay)

    # GA-Conv for feature encoding
    l3_xyz, l3_points, l3_indices = pointnet_GAconv(c1_l2_xyz, fc_l2_points, npoint=64, radius=0.4, nsample=32, mlp=[512,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')

    # Up-sampling, i.e., FP module
    l21_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [512,512], is_training, bn_decay, scope='fa_layer2')
    _, l2_points_1, _ = pointSIFT_module(l2_xyz, l21_points, radius=0.5, out_channel=512, is_training=is_training, bn_decay=bn_decay, scope='fa_layer2_c0')
    _, l2_points_2, _ = pointSIFT_module(l2_xyz, l21_points, radius=0.5, out_channel=512, is_training=is_training, bn_decay=bn_decay, scope='fa_layer2_c1')
    _, l2_points_3, _ = pointSIFT_module(l2_xyz, l21_points, radius=0.5, out_channel=512, is_training=is_training, bn_decay=bn_decay, scope='fa_layer2_c2')
    l21_points = tf.concat([l2_points_1, l2_points_2, l2_points_3], axis=-1)
    l21_points = tf_util.conv1d(l21_points, 512, 1, padding='VALID', bn=True, is_training=is_training, scope='fa_2_fc', bn_decay=bn_decay)

    # Nested architecture
    l21_points=tf.concat([fc_l2_points,l21_points],axis=-1) # 512+512=1024
    l21_points=tf_util.conv1d(l21_points, 512, 1, padding='VALID', bn=True, is_training=is_training, scope='l21', bn_decay=bn_decay)

    # PointSIFT embedding
    l12_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l21_points, [256,256], is_training, bn_decay, scope='fa_layer3')
    _, l1_points_1, _ = pointSIFT_module(l1_xyz, l12_points, radius=0.25, out_channel=256, is_training=is_training, bn_decay=bn_decay, scope='fa_layer3_c0')
    _, l1_points_2, _ = pointSIFT_module(l1_xyz, l1_points_1, radius=0.25, out_channel=256, is_training=is_training, bn_decay=bn_decay, scope='fa_layer3_c1')
    l12_points = tf.concat([l1_points_1, l1_points_2], axis=-1)
    l12_points = tf_util.conv1d(l12_points, 256, 1, padding='VALID', bn=True, is_training=is_training, scope='fa_1_fc', bn_decay=bn_decay)

    # Nested architecture
    l12_points = tf.concat([l12_points,l11_points,c0_l1_points],axis=-1)
    l12_points = tf_util.conv1d(l12_points, 256, 1, padding='VALID', bn=True, is_training=is_training, scope='l12',bn_decay=bn_decay)

    # PointSIFT embedding
    l03_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l12_points, [128,128,128], is_training, bn_decay, scope='fa_layer4')
    _, l03_points, _ = pointSIFT_module(l0_xyz, l03_points, radius=0.1, out_channel=128, is_training=is_training, bn_decay=bn_decay, scope='fa_layer4_c0')

    # Nested architecture
    l03_points= tf.concat([l03_points, l02_points, l01_points,c0_l0_points], axis=-1)
    l03_points = tf_util.conv1d(l03_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='l03',bn_decay=bn_decay)

    l03_points = tf.multiply(l0_z_weight, l03_points)

    # FC layers & deep supervision
    net3 = tf_util.conv1d(l03_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net3 = tf_util.dropout(net3, keep_prob=0.5, is_training=is_training, scope='dp3')
    net3 = tf_util.conv1d(net3, num_class, 1, padding='VALID', activation_fn=None, scope='fc2')

    net2 = tf_util.conv1d(l02_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc12', bn_decay=bn_decay)
    net2 = tf_util.dropout(net2, keep_prob=0.5, is_training=is_training, scope='dp2')
    net2 = tf_util.conv1d(net2, num_class, 1, padding='VALID', activation_fn=None, scope='fc22')

    net1 = tf_util.conv1d(l01_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc13', bn_decay=bn_decay)
    net1 = tf_util.dropout(net1, keep_prob=0.5, is_training=is_training, scope='dp1')
    net1 = tf_util.conv1d(net1, num_class, 1, padding='VALID', activation_fn=None, scope='fc23')

    return [net1, net2, net3, (net1+net2+net3)/3], end_points


def get_loss(pred, label, smpw):
    """
    :param pred: BxNxC
    :param label: BxN
    :param smpw: BxN
    :return:
    """
    # Deep supervision
    loss0 = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred[0], weights=smpw)
    loss1 = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred[1], weights=smpw)
    loss2 = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred[2], weights=smpw)
    classify_loss = (loss0+ loss1 + loss2)/3

    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss

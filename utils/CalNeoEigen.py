import tensorflow as tf
import numpy as np
tf.enable_eager_execution()
import math
import time
def compute_determinant(A):  # 行列式
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
    eig2 = 3 * q - eig1 - eig3
    return tf.abs(tf.stack([eig1, eig2, eig3], axis=2))  # (N, P, 3)

def get_neigh_geo_feat(new_xyz, grouped_xyz, nsample):
    grouped_xyz_mean = tf.reduce_mean(grouped_xyz, 2)
    grouped_xyz_offset = grouped_xyz - tf.tile(tf.expand_dims(grouped_xyz_mean, 2), [1, 1, nsample, 1])
    grouped_xyz_offset1 = tf.expand_dims(grouped_xyz_offset, -1)
    grouped_xyz_offset2 = tf.expand_dims(grouped_xyz_offset, 3)
    cov_in = tf.multiply(grouped_xyz_offset1, grouped_xyz_offset2)
    cov_sum = tf.reduce_sum(cov_in, 2)
    cov_out = tf.divide(cov_sum, nsample)


    eigenvalues, eigenvectors = tf.self_adjoint_eig(cov_out)

    e_sum = tf.tile(tf.expand_dims(tf.reduce_sum(eigenvalues, 2), 2), [1, 1, 3])
    eigenvalues = tf.div(eigenvalues, e_sum)  # Eigenvalues are normalised to sum up to 1

    lamba1 = tf.slice(eigenvalues, [0, 0, 2], [-1, -1, 1])
    lamba2 = tf.slice(eigenvalues, [0, 0, 1], [-1, -1, 1])
    lamba3 = tf.slice(eigenvalues, [0, 0, 0], [-1, -1, 1])
    e1 = tf.slice(eigenvectors, [0, 0, 2, 0], [-1, -1, 1, -1])
    e2 = tf.slice(eigenvectors, [0, 0, 1, 0], [-1, -1, 1, -1])
    e3 = tf.slice(eigenvectors, [0, 0, 0, 0], [-1, -1, 1, -1])
    feat_Sum = lamba1 + lamba2 + lamba3
    feat_Omnivariance = tf.pow((lamba1 * lamba2 * lamba3), 1 / 3)
    feat_Eigenentropy = tf.negative(lamba1 * tf.log(lamba1) + lamba2 * tf.log(lamba2) + lamba3 * tf.log(lamba3))
    feat_Anisotropy = (lamba1 - lamba3) / lamba1
    feat_Planarity = (lamba2 - lamba3) / lamba1
    feat_Linearity = (lamba1 - lamba2) / lamba1
    feat_Surface_Variation = lamba3 / (lamba1 + lamba2 + lamba3)
    feat_Sphericity = lamba3 / lamba1
    feat_Verticality = 1 - tf.abs(tf.squeeze(tf.slice(e3, [0, 0, 0, 2], [-1, -1, -1, 1]), axis=[3]))
    # Moment
    grouped_xyz_moment = grouped_xyz - tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])
    e1_tile = tf.tile(e1, [1, 1, nsample, 1])
    e2_tile = tf.tile(e2, [1, 1, nsample, 1])

    moment1 = tf.expand_dims(tf.reduce_sum(tf.reduce_sum(tf.multiply(grouped_xyz_moment, e1_tile),3),2),-1)
    moment2 = tf.expand_dims(tf.reduce_sum(tf.reduce_sum(tf.multiply(grouped_xyz_moment, e2_tile),3),2),-1)

    moment3 = tf.expand_dims(tf.reduce_sum(tf.square(tf.reduce_sum(tf.multiply(grouped_xyz_moment, e1_tile), 3)), 2), 2)
    moment4 = tf.expand_dims(tf.reduce_sum(tf.square(tf.reduce_sum(tf.multiply(grouped_xyz_moment, e2_tile), 3)), 2), 2)

    zmax = tf.reduce_max(grouped_xyz, [2])
    zmax = tf.slice(zmax, [0, 0, 2], [-1, -1, 1])
    zmax_tile = tf.tile(tf.expand_dims(zmax, -1), [1, 1, nsample, 1])
    zmin = tf.reduce_min(grouped_xyz, [2])
    zmin = tf.slice(zmin, [0, 0, 2], [-1, -1, 1])
    zmin_tile = tf.tile(tf.expand_dims(zmin, -1), [1, 1, nsample, 1])
    zmax_zmin = zmax - zmin

    z = tf.slice(grouped_xyz, [0, 0, 0, 2], [-1, -1, -1, 1])
    zmax_z = zmax_tile - z  # (16,1024,32,1)
    z_zmin = z - zmin_tile  # (16,1024,32,1)

    neigh_geofeat = tf.concat([feat_Sum, feat_Omnivariance, feat_Eigenentropy, feat_Anisotropy, feat_Planarity,
                               feat_Linearity, feat_Surface_Variation, feat_Sphericity, feat_Verticality, moment1,
                               moment2, moment3, moment4, zmax_zmin], axis=2)
    return neigh_geofeat, zmax_z ,z_zmin

if __name__=='__main__':
    grouped_xyz = tf.constant(np.random.rand(16, 1024, 8, 3),dtype = tf.float32)
    new_xyz = tf.constant(np.random.rand(16, 1024, 3),dtype = tf.float32)

    # neigh_geo_feat,z1,z2=get_neigh_geo_feat(new_xyz,grouped_xyz,8)
    # print(neigh_geo_feat)
    # print(z1)
    # print(z2)
    grouped_xyz_mean = tf.reduce_mean(grouped_xyz, 2)
    grouped_xyz_offset = grouped_xyz - tf.tile(tf.expand_dims(grouped_xyz_mean, 2), [1, 1, 8, 1])
    grouped_xyz_offset1 = tf.expand_dims(grouped_xyz_offset, -1)
    grouped_xyz_offset2 = tf.expand_dims(grouped_xyz_offset, 3)
    cov_in = tf.multiply(grouped_xyz_offset1, grouped_xyz_offset2)
    cov_sum = tf.reduce_sum(cov_in, 2)
    cov_out = tf.divide(cov_sum, 8)
    now=time.time()
    eigenvalues, eigenvectors = tf.self_adjoint_eig(cov_out)  # sorted from  min to max
    print(time.time()-now)
    now1=time.time()
    eigenvalues1=compute_eigenvals(cov_out)  # sorted from  max to min
    print(time.time()-now1)
    print(eigenvalues[0,0,:])
    print(eigenvalues1[0,0,:])




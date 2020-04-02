from __future__ import division
from __future__ import print_function


import argparse
import math
from datetime import datetime
import time
import numpy as np
import tensorflow as tf
import importlib
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # dfc
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(ROOT_DIR, 'models')) # no, really model
sys.path.append(ROOT_DIR) # provider
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
import pc_util
import dfc_dataset1

# train the model by block_data

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,5"

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--gpu_num', type=int, default=1, help='number of GPU to train')
parser.add_argument('--model', default='pointnet2_sem_seg', help='Model name [default: pointnet2_sem_seg]')
parser.add_argument('--data_dir', default=os.path.join(ROOT_DIR,'data','dfc','train'),help='Path to training dataset directory')
parser.add_argument('--log_dir', default='log/dfc', help='Log dir [default: log/dfc]')
parser.add_argument('--num_point', type=int, default=4096, help='Point Number [default: 8192]')
parser.add_argument('--max_epoch', type=int, default=1000, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=14, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=31840, help='Decay step for lr decay [default: 68720 (=40x number of training point clouds after split)]')
# origin set as :3280
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--existing-model', default='', help='Path to existing model to continue to train on')
parser.add_argument('--starting-epoch', type=int, default=0, help='Initial epoch number (for use when reloading model')
parser.add_argument('--extra-dims', type=int, default=[], nargs='*', help='Extra dims')
parser.add_argument('--log-weighting', dest='log_weighting', action='store_true')
parser.add_argument('--no-log-weighting', dest='log_weighting', action='store_false')
parser.set_defaults(log_weighting=True)  # remember to modify~
FLAGS = parser.parse_args()

EPOCH_CNT = 0

# basic params..
NUM_POINT = FLAGS.num_point
BATCH_SZ = FLAGS.batch_size
LEARNING_RATE = FLAGS.learning_rate
MAX_EPOCH = FLAGS.max_epoch
GPU_INDEX = FLAGS.gpu
GPU_NUM = FLAGS.gpu_num
BATCH_PER_GPU = BATCH_SZ // GPU_NUM

NUM_CLASS = 9

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp '+__file__+' %s' % (LOG_DIR)) # bkp of train procedure
os.system('cp %s %s' % (os.path.join(FLAGS.data_dir,'dfc_train_metadata.pickle'),LOG_DIR)) # copy of training metadata
LOG_FOUT = open(os.path.join(LOG_DIR, 'train.log'), 'w')  # write in train.log
LOG_FOUT.write(str(FLAGS)+'\n')
# lr params..
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

# bn params..
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

class SegTrainer(object):
    def __init__(self):
        self.train_data = None
        self.val_data=None
        self.test_data = None
        self.train_sz = 0
        self.val_sz = 0
        self.test_sz = 0
        self.point_sz = NUM_POINT   # 4096

        # batch loader init....
        self.batch_loader = None
        self.batch_sz = BATCH_SZ

        # net param...
        self.point_pl = None
        self.label_pl = None
        self.smpws_pl = None
        self.is_train_pl = None
        self.ave_tp_pl = None
        self.net = None
        self.end_point = None
        self.bn_decay = None
        self.loss = None
        self.optimizer = None
        self.train_op = None
        self.predict = None
        self.TP = None
        self.batch = None  # record the training step..

        # summary
        self.ave_tp_summary = None

        # list for multi gpu tower..
        self.tower_grads = []
        self.net_gpu = []
        self.total_loss_gpu_list = []

    def load_data(self):
        assert os.path.exists(FLAGS.data_dir), 'train_data not found !!!'
        self.train_data = dfc_dataset1.DFCDataset(root=FLAGS.data_dir, npoints=self.point_sz, split='train',log_weighting=FLAGS.log_weighting,extra_features=FLAGS.extra_dims)
        self.val_data = dfc_dataset1.DFCDataset(root=FLAGS.data_dir, npoints=self.point_sz, split='val',extra_features=FLAGS.extra_dims)
        self.test_data = dfc_dataset1.DFCDataset(root=FLAGS.data_dir, npoints=self.point_sz, split='test',extra_features=FLAGS.extra_dims)
        self.train_sz = self.train_data.__len__()
        self.val_sz = self.val_data.__len__()
        self.test_sz = self.test_data.__len__()

        print('train size %d and val size %d and test size %d' % (self.train_sz, self.val_sz, self.test_sz))

    def log_string(self,out_str):
        LOG_FOUT.write(out_str + '\n')
        LOG_FOUT.flush()
        print(out_str)


    def get_learning_rate(self):
        # learning_rate = tf.train.exponential_decay(LEARNING_RATE,
        #                                            self.batch * BATCH_SZ,
        #                                            DECAY_STEP,
        #                                            DECAY_RATE,
        #                                            staircase=True)
        learning_rate = tf.train.cosine_decay_restarts(LEARNING_RATE,self.batch * BATCH_SZ, 15920, t_mul=2.0,m_mul= 1.0,alpha=0.0)
        learning_rate = tf.maximum(learning_rate, 1e-5)
        tf.summary.scalar('learning rate', learning_rate)
        return learning_rate

    def get_bn_decay(self):
        bn_momentum = tf.train.exponential_decay(BN_INIT_DECAY,
                                                 self.batch * BATCH_SZ,
                                                 BN_DECAY_DECAY_STEP,
                                                 BN_DECAY_DECAY_RATE,
                                                 staircase=True)
        bn_momentum = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
        tf.summary.scalar('bn_decay', bn_momentum)
        return bn_momentum

    def get_batch_wdp(self, dataset, idxs, start_idx, end_idx):
        bsize = end_idx - start_idx
        batch_data = np.zeros((bsize, self.point_sz, len(self.train_data.columns)))
        batch_label = np.zeros((bsize, self.point_sz), dtype=np.int32)
        batch_smpw = np.zeros((bsize, self.point_sz), dtype=np.float32)
        for i in range(bsize):
            if start_idx + i < len(dataset):
                ps, seg, smpw = dataset[idxs[i + start_idx]]
                batch_data[i, ...] = ps
                batch_label[i, :] = seg
                batch_smpw[i, :] = smpw

                dropout_ratio = np.random.random() * 0.875  # 0-0.875
                drop_idx = np.where(np.random.random((ps.shape[0])) <= dropout_ratio)[0]
                batch_data[i, drop_idx, :] = batch_data[i, 0, :]
                batch_label[i, drop_idx] = batch_label[i, 0]
                batch_smpw[i, drop_idx] *= 0
        return batch_data, batch_label, batch_smpw

    def get_batch(self, dataset, idxs, start_idx, end_idx):
        bsize = end_idx - start_idx
        batch_data = np.zeros((bsize, self.point_sz, len(self.train_data.columns)))
        batch_label = np.zeros((bsize, self.point_sz), dtype=np.int32)
        batch_smpw = np.zeros((bsize, self.point_sz), dtype=np.float32)
        for i in range(bsize):
            if start_idx + i < len(dataset):
                ps, seg, smpw = dataset[idxs[i + start_idx]]
                batch_data[i, ...] = ps
                batch_label[i, :] = seg
                batch_smpw[i, :] = smpw
        return batch_data, batch_label, batch_smpw

    @staticmethod
    def ave_gradient(tower_grad):
        ave_gradient = []
        for gpu_data in zip(*tower_grad):
            grads = []
            for g, k in gpu_data:
                t_g = tf.expand_dims(g, axis=0)
                grads.append(t_g)
            grad = tf.concat(grads, axis=0)
            grad = tf.reduce_mean(grad, axis=0)
            key = gpu_data[0][1]
            ave_gradient.append((grad, key))
        return ave_gradient

    # cpu part of graph
    def build_g_cpu(self):
        self.batch = tf.Variable(0, name='batch', trainable=False)
        self.point_pl, self.label_pl, self.smpws_pl = MODEL.placeholder_inputs(self.batch_sz, self.point_sz)
        self.is_train_pl = tf.placeholder(dtype=tf.bool, shape=())
        self.ave_tp_pl = tf.placeholder(dtype=tf.float32, shape=())
        self.optimizer = tf.train.AdamOptimizer(self.get_learning_rate())
        self.bn_decay = self.get_bn_decay()

        MODEL.get_model(self.point_pl, self.is_train_pl, num_class=NUM_CLASS, bn_decay=self.bn_decay)

    # graph for each gpu, reuse params...
    def build_g_gpu(self, gpu_idx):
        print("build graph in gpu %d" % gpu_idx)
        with tf.device('/gpu:%d' % gpu_idx), tf.name_scope('gpu_%d' % gpu_idx) as scope:
            point_cloud_slice = tf.slice(self.point_pl, [gpu_idx * BATCH_PER_GPU, 0, 0], [BATCH_PER_GPU, -1, -1])
            label_slice = tf.slice(self.label_pl, [gpu_idx * BATCH_PER_GPU, 0], [BATCH_PER_GPU, -1])
            smpws_slice = tf.slice(self.smpws_pl, [gpu_idx * BATCH_PER_GPU, 0], [BATCH_PER_GPU, -1])
            net, end_point = MODEL.get_model(point_cloud_slice, self.is_train_pl, num_class=NUM_CLASS,
                                                 bn_decay=self.bn_decay)
            MODEL.get_loss(net, label_slice, smpw=smpws_slice)
            loss = tf.get_collection('losses', scope=scope)
            total_loss = tf.add_n(loss, name='total_loss')
            for _i in loss + [total_loss]:
                tf.summary.scalar(_i.op.name, _i)

            gvs = self.optimizer.compute_gradients(total_loss)
            self.tower_grads.append(gvs)
            self.net_gpu.append(net[2])  # net[2], with deep supervision
            self.total_loss_gpu_list.append(total_loss)

    def build_graph(self):
        with tf.device('/cpu:0'):
            self.build_g_cpu()
            self.tower_grads = []
            self.net_gpu = []
            self.total_loss_gpu_list = []

            for i in range(GPU_NUM):
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    self.build_g_gpu(i)

            self.net = tf.concat(self.net_gpu, axis=0)
            self.loss = tf.reduce_mean(self.total_loss_gpu_list)

            # get training op
            gvs = self.ave_gradient(self.tower_grads)
            self.train_op = self.optimizer.apply_gradients(gvs, global_step=self.batch)
            self.predict = tf.cast(tf.argmax(self.net, axis=2), tf.int32)
            # TP is accuracy in fact.
            self.TP = tf.reduce_sum(
                tf.cast(tf.equal(self.predict, self.label_pl), tf.float32)) / self.batch_sz / self.point_sz
            tf.summary.scalar('TP', self.TP)
            tf.summary.scalar('total_loss', self.loss)

    def training(self):
        with tf.Graph().as_default():
            self.build_graph()
            # merge operator (for tensorboard)
            merged = tf.summary.merge_all()
            # iter_in_epoch = self.train_sz // self.batch_sz
            num_batches = int(math.ceil((1.0 * self.train_sz)/self.batch_sz))


            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
            bestsaver = tf.train.Saver(max_to_keep=20)

            # Create a session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False

            best_acc = 0.0
            best_mIoU= 0.0
            with tf.Session(config=config) as sess:
                if FLAGS.existing_model:
                    self.log_string("Loading model from " + FLAGS.existing_model)
                    saver.restore(sess, FLAGS.existing_model)
                train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
                val_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'val'), sess.graph)
                test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

                # Init variables
                #sess.run(tf.global_variables_initializer())
                if not FLAGS.existing_model:
                    init = tf.global_variables_initializer()
                    sess.run(init)
                epoch_sz = MAX_EPOCH

                ops = {'pointclouds_pl': self.point_pl,
                       'labels_pl':self.label_pl,
                       'smpws_pl': self.smpws_pl,
                       'is_training_pl':self.is_train_pl,
                       'pred': self.predict,
                       'loss': self.loss,
                       'train_op': self.train_op,
                       'merged': merged,
                       'step': self.batch,
                       'net':self.net,
                       'end_points': self.end_point}

                for epoch in range(FLAGS.starting_epoch,epoch_sz):
                    self.log_string('**** EPOCH%3d ****' % (epoch))
                    sys.stdout.flush()

                    self.train_one_epoch(sess,ops,train_writer)

                    do_save = epoch % 10 == 0
                    if epoch % 1 == 0:
                        self.log_string(str(datetime.now()))
                        self.log_string('---- EPOCH %03d EVALUATION ----' % (epoch))
                        acc1 , mIoU1= self.eval_one_epoch(sess, ops,val_writer)
                        acc, mIoU= self.eval_onetest_epoch(sess, ops,test_writer)

                        if acc < best_acc and (best_acc-acc) < 0.002:
                            save_path = bestsaver.save(sess, os.path.join(LOG_DIR,"best_model_acc2_%d.ckpt" % (epoch*self.train_sz)))
                            self.log_string("Model saved in file: %s" % save_path)
                        if acc > best_acc:
                            save_path = bestsaver.save(sess, os.path.join(LOG_DIR,"best_model_acc_%d.ckpt" % (epoch*self.train_sz)))
                            self.log_string("Model saved in file: %s" % save_path)
                            best_acc = acc
                            do_save = False
                        if mIoU > best_mIoU:
                            save_path = bestsaver.save(sess, os.path.join(LOG_DIR, "best_model_mIoU_%d.ckpt" % ( epoch * self.train_sz)))
                            self.log_string("Model saved in file: %s" % save_path)
                            best_mIoU = mIoU
                            do_save = False
                            # Save the variables to disk.
                    if do_save:
                        save_path = saver.save(sess, os.path.join(LOG_DIR, "model_%d.ckpt" % (epoch*self.train_sz)))
                        self.log_string("Model saved in file: %s" % save_path)

    def train_one_epoch(self, sess, ops, train_writer):
        """ ops: dict mapping from string to tf ops """
        is_training = True

        # Shuffle train samples
        train_idxs = np.arange(0, self.train_data.__len__())   # train_idx ,idx is a list [1263, 396,1309 .... 41 398 495]
        np.random.shuffle(train_idxs)
        num_batches = int(math.ceil((1.0 * self.train_sz) / self.batch_sz))

        self.log_string(str(datetime.now()))

        total_correct = 0
        total_seen = 0
        loss_sum = 0
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_sz
            end_idx = (batch_idx + 1) * self.batch_sz
            batch_data, batch_label, batch_smpw = self.get_batch_wdp(self.train_data, train_idxs,
                                                                     start_idx, end_idx)
            # Augment batched point clouds by rotation and jitter
            if FLAGS.extra_dims:
                aug_data = np.concatenate((provider.rotate_point_cloud_z(batch_data[:, :, 0:3]),
                                           batch_data[:, :, 3:]), axis=2)
                aug_data = np.concatenate((provider.jitter_point_cloud(aug_data[:, :, 0:3]),
                                           aug_data[:, :, 3:]), axis=2)
            else:
                aug_data = provider.rotate_point_cloud_z(batch_data)
                aug_data=provider.jitter_point_cloud(aug_data)

            feed_dict = {ops['pointclouds_pl']: aug_data,
                         ops['labels_pl']: batch_label,
                         ops['smpws_pl']: batch_smpw,
                         ops['is_training_pl']: is_training, }
            summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                             ops['train_op'], ops['loss'], ops['pred']],
                                                            feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
            # pred_val = np.argmax(pred_val, 2)
            correct = np.sum(pred_val == batch_label)
            total_correct += correct
            total_seen += (self.batch_sz * self.point_sz)
            loss_sum += loss_val
            if (batch_idx + 1) % 5 == 0:
                self.log_string(' -- %03d / %03d --' % (batch_idx + 1, num_batches))
                self.log_string('mean loss: %f' % (loss_sum / 5))
                self.log_string('accuracy: %f' % (total_correct / float(total_seen)))
                total_correct = 0
                total_seen = 0
                loss_sum = 0

    # evaluate on randomly chopped scenes
    def eval_one_epoch(self, sess, ops, val_writer):
        """ ops: dict mapping from string to tf ops """
        is_training = False
        val_idxs = np.arange(0,self.val_sz)
        num_batches = int(math.ceil((1.0 * self.val_sz) / self.batch_sz))

        total_correct = 0
        total_seen = 0
        loss_sum = 0
        total_seen_class = [0 for _ in range(NUM_CLASS)]
        total_correct_class = [0 for _ in range(NUM_CLASS)]

        labelweights = np.zeros(NUM_CLASS)
        tp = np.zeros(NUM_CLASS)
        fp = np.zeros(NUM_CLASS)
        fn = np.zeros(NUM_CLASS)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_sz
            end_idx = (batch_idx + 1) * self.batch_sz
            batch_data, batch_label, batch_smpw = self.get_batch(self.val_data, val_idxs, start_idx, end_idx)

            if FLAGS.extra_dims:
                aug_data = np.concatenate((provider.rotate_point_cloud_z(batch_data[:, :, 0:3]),
                                           batch_data[:, :, 3:]), axis=2)
                aug_data = np.concatenate((provider.jitter_point_cloud(aug_data[:, :, 0:3]),
                                           aug_data[:, :, 3:]), axis=2)
            else:
                aug_data = provider.rotate_point_cloud_z(batch_data)
                aug_data = provider.jitter_point_cloud(aug_data)


            feed_dict = {ops['pointclouds_pl']: aug_data,
                         ops['labels_pl']: batch_label,
                         ops['smpws_pl']: batch_smpw,
                         ops['is_training_pl']: is_training}
            summary, step, loss_val, net = sess.run([ops['merged'], ops['step'],
                                                          ops['loss'], ops['net']], feed_dict=feed_dict)
            val_writer.add_summary(summary, step)

            pred_val = np.argmax(net, axis = 2)  # BxN
            correct = np.sum((pred_val == batch_label) & (batch_label > 0) & (batch_smpw > 0))  # evaluate only all categories except unknown
            total_correct += correct
            total_seen += np.sum((batch_label > 0) & (batch_smpw > 0))
            loss_sum += loss_val

            for l in range(NUM_CLASS):
                total_seen_class[l] += np.sum((batch_label == l) & (batch_smpw > 0))
                total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l) & (batch_smpw > 0))
                tp[l] += ((pred_val == l) & (batch_label == l)).sum()
                fp[l] += ((pred_val == l) & (batch_label != l)).sum()
                fn[l] += ((pred_val != l) & (batch_label == l)).sum()

        self.log_string('val_eval mean loss: %f' % (loss_sum / float(num_batches)))
        self.log_string('val_eval point accuracy: %f' % (total_correct / float(total_seen)))
        self.log_string('val_eval point avg class acc: %f' % (
            np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
        per_class_str = '     '
        iou = np.divide(tp, tp + fp + fn)
        for l in range(NUM_CLASS):
            per_class_str += 'class %d[%d] acc: %f, iou: %f; ' % (
            self.val_data.decompress_label_map[l], l, total_correct_class[l] / float(total_seen_class[l]), iou[l])
        self.log_string(per_class_str)
        self.log_string('val_mIOU: {}'.format(iou.mean()))
        acc = total_correct / float(total_seen)
        mIoU=iou.mean()

        return acc , mIoU

    # evaluate on randomly chopped scenes
    def eval_onetest_epoch(self, sess, ops, test_writer):
        """ ops: dict mapping from string to tf ops """
        is_training = False
        test_idxs = np.arange(0,self.test_sz)
        num_batches = int(math.ceil((1.0 * self.test_sz) / self.batch_sz))

        total_correct = 0
        total_seen = 0
        loss_sum = 0
        total_seen_class = [0 for _ in range(NUM_CLASS)]
        total_correct_class = [0 for _ in range(NUM_CLASS)]

        labelweights = np.zeros(NUM_CLASS)
        tp = np.zeros(NUM_CLASS)
        fp = np.zeros(NUM_CLASS)
        fn = np.zeros(NUM_CLASS)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_sz
            end_idx = (batch_idx + 1) * self.batch_sz
            batch_data, batch_label, batch_smpw = self.get_batch(self.test_data, test_idxs, start_idx, end_idx)

            if FLAGS.extra_dims:
                aug_data = np.concatenate((provider.rotate_point_cloud_z(batch_data[:, :, 0:3]),
                                           batch_data[:, :, 3:]), axis=2)
                aug_data = np.concatenate((provider.jitter_point_cloud(aug_data[:, :, 0:3]),
                                           aug_data[:, :, 3:]), axis=2)
            else:
                aug_data = provider.rotate_point_cloud_z(batch_data)
                aug_data = provider.jitter_point_cloud(aug_data)


            feed_dict = {ops['pointclouds_pl']: aug_data,
                         ops['labels_pl']: batch_label,
                         ops['smpws_pl']: batch_smpw,
                         ops['is_training_pl']: is_training}
            summary, step, loss_val, net = sess.run([ops['merged'], ops['step'],
                                                          ops['loss'], ops['net']], feed_dict=feed_dict)
            test_writer.add_summary(summary, step)

            pred_val = np.argmax(net, axis = 2)  # BxN
            correct = np.sum((pred_val == batch_label) & (batch_label > 0) & (batch_smpw > 0))  # evaluate only all categories except unknown
            total_correct += correct
            total_seen += np.sum((batch_label > 0) & (batch_smpw > 0))
            loss_sum += loss_val

            for l in range(NUM_CLASS):
                total_seen_class[l] += np.sum((batch_label == l) & (batch_smpw > 0))
                total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l) & (batch_smpw > 0))
                tp[l] += ((pred_val == l) & (batch_label == l)).sum()
                fp[l] += ((pred_val == l) & (batch_label != l)).sum()
                fn[l] += ((pred_val != l) & (batch_label == l)).sum()

        self.log_string('eval test_mean loss: %f' % (loss_sum / float(num_batches)))
        self.log_string('eval test_point accuracy: %f' % (total_correct / float(total_seen)))
        self.log_string('eval test_point avg class acc: %f' % (
            np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
        avg_acc=np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))
        per_class_str = '     '
        iou = np.divide(tp, tp + fp + fn)
        for l in range(NUM_CLASS):
            per_class_str += 'class %d[%d] acc: %f, iou: %f; ' % (
            self.test_data.decompress_label_map[l], l, total_correct_class[l] / float(total_seen_class[l]), iou[l])
        self.log_string(per_class_str)
        self.log_string('test_mIOU: {}'.format(iou.mean()))
        acc = total_correct / float(total_seen)
        mIoU=iou.mean()

        return acc , avg_acc


if __name__ == '__main__':
    trainer = SegTrainer()
    trainer.load_data()
    trainer.training()


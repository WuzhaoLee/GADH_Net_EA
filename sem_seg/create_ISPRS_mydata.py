import argparse
import glob
import logging
import multiprocessing
import numpy as np
import os
import pickle
import pprint
import random
import sys

from pointset1 import PointSet


def get_list_of_files(folder):
    files = glob.glob(os.path.join(folder, "*_PC3.txt"))
    return sorted(files)


def load_file(pc3_path):
    logging.info("Loading '" + pc3_path + "'")

    cls_path = pc3_path[:-7] + 'CLS.txt'

    pset = PointSet(pc3_path, cls_path)

    psets = pset.split()

    return [extract_block_data(ps) for ps in psets]  # ps is a PointSet Object


def extract_block_data(data,label):  # for each block
    #data64 = np.stack([pset.x, pset.y, pset.z, pset.i, pset.r], axis=1)
    offsets = np.mean(data, axis=0)
    variances = np.var(data, axis=0)
    data = (data-offsets).astype('float32')

    chist = {}
    for C in label:
        if C in chist:
            chist[C] += 1
        else:
            chist[C] = 1

    metadata = {
        "offsets": offsets,
        "variance": variances,
        "N": data.shape[0],
        "class_count": chist}

    return [metadata, data, label]


def reduce_metadata(all_metadata):
    sum_N = 0
    cls_hist = {}
    sum_variances = np.zeros((5,))
    for d in all_metadata:
        sum_N += d['N']
        sum_variances += d['N'] * d['variance']
        for cls, cnt in d['class_count'].items():
            if cls in cls_hist:
                cls_hist[cls] += cnt
            else:
                cls_hist[cls] = cnt

    compressed_label_map = {}
    decompress_label_map = {}

    i = 0
    for key in sorted(cls_hist):
        compressed_label_map[key] = i
        decompress_label_map[i] = key
        i += 1

    metadata = {
        "variance": sum_variances / sum_N,
        "cls_hist": cls_hist,
        "compressed_label_map": compressed_label_map,
        "decompress_label_map": decompress_label_map}
    return metadata


def parse_args(argv):
    # Setup arguments & parse
    parser = argparse.ArgumentParser(
        description=__doc__,  # printed with -h/--help
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input-path', help='e.g. /path/to/DFC/Track4', required=True)
    parser.add_argument('-o', '--output-path', help='e.g. /path/to/training_data_folder', required=True)
    parser.add_argument('-f', '--training-frac', help='Fraction of data to use for training vs validation', default=0.5,
                        type=float)

    return parser.parse_args(argv[1:])


def start_log(opts):
    if not os.path.exists(opts.output_path):
        os.makedirs(opts.output_path)

    rootLogger = logging.getLogger()

    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-3.3s] %(message)s")
    fileHandler = logging.FileHandler(
        os.path.join(opts.output_path, os.path.splitext(os.path.basename(__file__))[0] + '.log'), mode='w')
    fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel(logging.DEBUG)
    rootLogger.addHandler(fileHandler)

    logFormatter = logging.Formatter("[%(levelname)-3.3s] %(message)s")
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    consoleHandler.setLevel(logging.INFO)
    rootLogger.addHandler(consoleHandler)

    rootLogger.level = logging.DEBUG

    logging.debug('Options:\n' + pprint.pformat(opts.__dict__))


def create_train_val(argv=None):
    opts = parse_args(argv if argv is not None else sys.argv)
    start_log(opts)

    pset = PointSet(os.path.join(opts.input_path,'Training_PC3.txt'), os.path.join(opts.input_path,'Training_CLS.txt'))
    [block_data_total, block_label_total] = pset.cloud2blocks_train()
    print(block_data_total.__len__())
    print(block_label_total.__len__())

    val_data_psets = np.concatenate((block_data_total[2], block_data_total[7]), axis=0)
    val_label_psets = np.concatenate((block_label_total[2], block_label_total[7]), axis=0)

    train_data_psets = np.concatenate((block_data_total[0], block_data_total[1], block_data_total[3],
                                       block_data_total[4], block_data_total[5], block_data_total[6],
                                       block_data_total[8], block_data_total[9], block_data_total[10],
                                       block_data_total[11], block_data_total[12]), axis=0)
    train_label_psets = np.concatenate((block_label_total[0], block_label_total[1], block_label_total[3],
                                        block_label_total[4], block_label_total[5], block_label_total[6],
                                        block_label_total[8], block_label_total[9], block_label_total[10],
                                        block_label_total[11], block_label_total[12]), axis=0)
    print(train_data_psets.__len__())
    print(val_data_psets.__len__())

    total_block = []
    for idx in range(train_data_psets.__len__()):
        block = extract_block_data(train_data_psets[idx], train_label_psets[idx])
        total_block.append(block)

    train_all_metadata = [block[0] for block in total_block]
    train_dataset = [block[1] for block in total_block]
    train_labels = [block[2] for block in total_block]
    train_metadata = reduce_metadata(train_all_metadata)

    val_total_block = []
    for idx in range(val_data_psets.__len__()):
        block = extract_block_data(val_data_psets[idx], val_label_psets[idx])
        val_total_block.append(block)

    val_all_metadata = [block[0] for block in val_total_block]
    val_dataset = [block[1] for block in val_total_block]
    val_labels = [block[2] for block in val_total_block]
    val_metadata = reduce_metadata(val_all_metadata)

    with open(os.path.join(opts.output_path, "dfc_" + 'train' + "_metadata.pickle"), 'wb') as f:
        pickle.dump(train_metadata, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(opts.output_path, "dfc_" + 'train' + "_dataset.pickle"), 'wb') as f:
        pickle.dump(train_dataset, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(opts.output_path, "dfc_" + 'train' + "_labels.pickle"), 'wb') as f:
        pickle.dump(train_labels, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(opts.output_path, "dfc_" + 'val' + "_metadata.pickle"), 'wb') as f:
        pickle.dump(val_metadata, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(opts.output_path, "dfc_" + 'val' + "_dataset.pickle"), 'wb') as f:
        pickle.dump(val_dataset, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(opts.output_path, "dfc_" + 'val' + "_labels.pickle"), 'wb') as f:
        pickle.dump(val_labels, f, pickle.HIGHEST_PROTOCOL)

def create_test(argv=None):
    opts = parse_args(argv if argv is not None else sys.argv)
    start_log(opts)

    pset = PointSet(os.path.join(opts.input_path,'EVAL_PC3.txt'), os.path.join(opts.input_path,'EVAL_CLS.txt'))
    [block_data_total, block_label_total] = pset.cloud2blocks_finaltest()
    print(block_data_total.__len__())
    print(block_label_total.__len__())
    test_data_psets=np.concatenate((block_data_total[0],block_data_total[1],block_data_total[2],block_data_total[3],block_data_total[4],block_data_total[5],
                                     block_data_total[6],block_data_total[7],block_data_total[8],block_data_total[9]),axis=0)
    test_label_psets=np.concatenate((block_label_total[0],block_label_total[1],block_label_total[2],block_label_total[3],block_label_total[4],block_label_total[5],
                                     block_label_total[6],block_label_total[7],block_label_total[8],block_label_total[9]),axis=0)
    print(test_data_psets.__len__())
    print(test_label_psets.__len__())

    total_block=[]
    for idx in range(test_data_psets.__len__()):
        block=extract_block_data(test_data_psets[idx],test_label_psets[idx])
        total_block.append(block)

    test_all_metadata = [block[0] for block in total_block]
    test_dataset=[block[1] for block in total_block]
    test_labels=[block[2] for block in total_block]
    test_metadata=reduce_metadata(test_all_metadata)

    with open(os.path.join(opts.output_path, "dfc_" + 'test' + "_metadata.pickle"), 'wb') as f:
        pickle.dump(test_metadata, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(opts.output_path, "dfc_" + 'test' + "_dataset.pickle"), 'wb') as f:
        pickle.dump(test_dataset, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(opts.output_path, "dfc_" + 'test' + "_labels.pickle"), 'wb') as f:
        pickle.dump(test_labels, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    create_train_val(sys.argv)
    create_test(sys.argv)
    print('Created')

















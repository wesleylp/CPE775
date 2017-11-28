import os
import argparse
import glob
import pandas as pd

from cpe775 import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', metavar='DIR')
    parser.add_argument('--out-dir', metavar='DIR')

    args = parser.parse_args()

    # Output to the data dir file if out dir was not set
    args.out_dir = args.out_dir or args.data_dir

    train_files = glob.glob(os.path.join(args.data_dir, 'helen', 'trainset', '**', '*.pts'), recursive=True)
    train_files += glob.glob(os.path.join(args.data_dir, 'lfpw', 'trainset', '**', '*.pts'), recursive=True)
    train_files += glob.glob(os.path.join(args.data_dir, 'afw', '**', '*.pts'), recursive=True)

    print('Found {} train files'.format(len(train_files)))

    train_set_fname = os.path.join(args.out_dir, 'train.csv')
    train_df = utils.read_pts(train_files, common_path=args.data_dir)
    train_df.to_csv(train_set_fname, header=None, index=False)

    print('Saving train set at {}'.format(os.path.abspath(train_set_fname)))

    common_test_files = glob.glob(os.path.join(args.data_dir, 'helen', 'testset', '**', '*.pts'), recursive=True)
    common_test_files += glob.glob(os.path.join(args.data_dir, 'lfpw', 'testset', '**', '*.pts'), recursive=True)

    print('Found {} files from the common subset of the 300W public test set'.format(len(common_test_files)))

    common_set_fname = os.path.join(args.out_dir, 'common_test.csv')
    common_df = utils.read_pts(common_test_files, common_path=args.data_dir)
    common_df.to_csv(common_set_fname, header=None, index=False)

    print('Saving common subset at {}'.format(os.path.abspath(common_set_fname)))

    challenging_test_files = glob.glob(os.path.join(args.data_dir, 'ibug', '**', '*.pts'), recursive=True)

    print('Found {} files from the challenging subset of the 300W public test set'.format(len(challenging_test_files)))

    challenging_set_fname = os.path.join(args.out_dir, 'challenging_test.csv')
    challenging_df = utils.read_pts(challenging_test_files, common_path=args.data_dir)
    challenging_df.to_csv(challenging_set_fname, header=None, index=False)

    print('Saving challenging subset at {}'.format(os.path.abspath(challenging_set_fname)))

    w300_test_files = glob.glob(os.path.join(args.data_dir, '300W', '0?_*', '**', '*.pts'), recursive=True)

    print('Found {} files from the 300W private test'.format(len(w300_test_files)))

    w300_set_fname = os.path.join(args.out_dir, 'w300_test.csv')
    w300_df = utils.read_pts(w300_test_files, common_path=args.data_dir)
    w300_df.to_csv(w300_set_fname, header=None, index=False)

    print('Saving w300 private set at {}'.format(os.path.abspath(w300_set_fname)))

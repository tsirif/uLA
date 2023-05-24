import os
from pathlib import Path
import itertools
from typing import Callable, Optional

import torch
import numpy
from PIL import Image
from torch.utils.data import Dataset


class MPI3D(Dataset):
    NUM_CLASSES = (6, 6, 2, 3, 3, 1, 1)  # `1` stands for regression
    NUM_CHANNELS = 3
    IMG_SIZE = 64
    MEAN = (0.1354, 0.1716, 0.1461)
    STD = (0.113, 0.117, 0.124)

    def __init__(self, data_path: Path,
            transform: Optional[Callable]=None,
            numpy_transform: Optional[Callable]=None,
            label_transform: bool=True):
        super(MPI3D, self).__init__()
        dataset = numpy.load(data_path)
        self.data = torch.from_numpy(numpy.asarray(dataset['data'], dtype=numpy.uint8))
        self.targets = numpy.asarray(dataset['attributes'], dtype=numpy.int64)
        self.transform = transform
        self.numpy_transform = numpy_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = self.data[index]
        if self.transform is not None:
            image = Image.fromarray(image.numpy())
            image = self.transform(image)
        elif self.numpy_transform is not None:
            image = self.numpy_transform(image.permute(2, 0, 1).contiguous())

        label = list(self.targets[index])
        label[0] = int(label[0])
        label[1] = int(label[1])
        label[2] = int(label[2])
        label[3] = int(label[3])
        label[4] = int(label[4])
        if self.label_transform:
            label[5] = 2 * (float(label[5]) / 39.) - 1  # in [-1, 1]
            label[6] = 2 * (float(label[6]) / 39.) - 1  # in [-1, 1]
        else:
            label[5] = int(label[5])
            label[6] = int(label[6])

        return tuple([image, ] + label)


def generate_dataset():
    import argparse
    import wget

    URL_BASE = "https://storage.googleapis.com/disentanglement_dataset/Final_Dataset"

    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", '-i', type=Path, default=os.path.curdir)
    parser.add_argument("--outpath", '-o', type=Path, default=None)
    parser.add_argument("--type", type=str, default="real")
    parser.add_argument('--confound', type=int, default=3,
        help="Lower integer means stronger confounding between color and shape in the training set")
    parser.add_argument("--train-size", type=int, default=180000)
    parser.add_argument("--valid-size", type=int, default=18000)
    parser.add_argument("--test-size", type=int, default=54000)
    parser.add_argument("--ffcv", action="store_true")
    parser.add_argument("--ffcv_probability", type=float, default=0.5)
    parser.add_argument("--ffcv_quality", type=int, default=90)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_workers', type=int,
        default=os.environ.get('SLURM_CPUS_PER_TASK', 4))
    parser.add_argument('--dry-run', action='store_true')

    args = parser.parse_args()

    base_filename = f"mpi3d_{args.type}.npz"
    filepath = args.datapath / base_filename
    print("Splitting dataset:", filepath)
    url = URL_BASE + "/" + base_filename

    outpath = args.outpath or args.datapath
    os.makedirs(outpath, exist_ok=True)

    if not os.path.isfile(filepath):
        print("Downloading", base_filename)
        wget.download(url, out=str(filepath))
    else:
        print("Found", base_filename)

    numpy.random.seed(args.seed)

    # Create targets
    print("Create targets")
    attributes = numpy.asarray([6, 6, 2, 3, 3, 40, 40])
    a = [slice(0, attr) for attr in attributes]
    data = numpy.mgrid[a]
    data = data.reshape(len(attributes), -1).T
    data = numpy.ascontiguousarray(data)

    # Create systematic splits according to targets
    K = args.confound
    assert K >= 1 and K < 6
    iid_idxs = (data[:, 1] == data[:, 0])
    for j in range(1, K):
        iid_idxs |= (data[:, 1] + j) % 6 == data[:, 0]
    ood_idxs = ~iid_idxs

    iid_combinations = numpy.zeros((6, 6), dtype=numpy.bool8)
    idxs_ = numpy.unique(data[iid_idxs][:, :2], axis=0)
    iid_combinations[idxs_[:, 0], idxs_[:, 1]] = True
    idxs_ = numpy.unique(data[ood_idxs][:, :2], axis=0)
    iid_combinations[idxs_[:, 0], idxs_[:, 1]] = False
    print('combinations:', iid_combinations)

    num_iid_combs = iid_combinations.sum()
    train_size = args.train_size // num_iid_combs
    valid_size = args.valid_size // num_iid_combs
    test_size = args.test_size // 6**2
    print(train_size, valid_size, test_size)

    idxs = numpy.arange(data.shape[0])
    train_idxs = []
    valid_idxs = []
    balanced_test_idxs = []
    ood_test_idxs = []
    for i, j in itertools.product(range(6), range(6)):
        idxs_ = idxs[(data[:, 0] == i) & (data[:, 1] == j)]
        numpy.random.shuffle(idxs_)
        if iid_combinations[i, j]:
            balanced_test_idxs.append(idxs_[:test_size])
            train_idxs.append(idxs_[test_size:test_size+train_size])
            valid_idxs.append(idxs_[test_size+train_size:test_size+train_size+valid_size])
        else:
            idxs_ = idxs_[:test_size]
            ood_test_idxs.append(idxs_)
            balanced_test_idxs.append(idxs_)
    train_idxs = numpy.concatenate(train_idxs)
    valid_idxs = numpy.concatenate(valid_idxs)
    balanced_test_idxs = numpy.concatenate(balanced_test_idxs)
    ood_test_idxs = numpy.concatenate(ood_test_idxs)

    print(len(train_idxs), len(valid_idxs), len(balanced_test_idxs), len(ood_test_idxs))
    if args.dry_run:
        return

    # Load raw dataset
    print("Loading raw dataset")
    zipfile = numpy.load(filepath,
        allow_pickle=True, encoding='bytes')
    imgs = zipfile['images'].reshape(6, 6, 2, 3, 3, 40, 40, 64, 64, 3)
    imgs = imgs.astype(numpy.uint8)

    # Shuffle first axis for different systematic splits
    print("Shuffle axes across color and shape")
    permute_color_axis = numpy.random.permutation(6)
    permute_shape_axis = numpy.random.permutation(6)
    imgs = imgs[permute_color_axis]
    imgs = imgs[:, permute_shape_axis]
    imgs = numpy.ascontiguousarray(imgs)
    imgs = imgs.reshape(-1, 64, 64, 3)

    print("Save training npz:", len(train_idxs))
    train_imgs = imgs[train_idxs]
    train_data = data[train_idxs]
    print('Training dataset stats:',
        numpy.mean(train_imgs[:50000], axis=tuple(range(3))) / 255.,
        numpy.std(train_imgs[:50000], axis=tuple(range(3))) / 255.)
    train_zipfile = {'data': train_imgs.astype(numpy.uint8),
                     'attributes': train_data.astype(numpy.uint8)}
    numpy.savez_compressed(
        outpath / f'mpi3d_{args.type}_train_K{K}_seed{args.seed}.npz',
        **train_zipfile)
    del train_imgs
    del train_data
    del train_zipfile

    print("Save valid npz:", len(valid_idxs))
    valid_imgs = imgs[valid_idxs]
    valid_data = data[valid_idxs]
    valid_zipfile = {'data': valid_imgs.astype(numpy.uint8),
                     'attributes': valid_data.astype(numpy.uint8)}
    numpy.savez_compressed(
        outpath / f'mpi3d_{args.type}_valid_K{K}_seed{args.seed}.npz',
        **valid_zipfile)
    del valid_imgs
    del valid_data
    del valid_zipfile

    print("Save test npz:", len(balanced_test_idxs))
    test_imgs = imgs[balanced_test_idxs]
    test_data = data[balanced_test_idxs]
    test_zipfile = {'data': test_imgs.astype(numpy.uint8),
                    'attributes': test_data.astype(numpy.uint8)}
    numpy.savez_compressed(
        outpath / f'mpi3d_{args.type}_test_K{K}_seed{args.seed}.npz',
        **test_zipfile)
    del test_imgs
    del test_data
    del test_zipfile

    print("Save ood npz:", len(ood_test_idxs))
    test_imgs = imgs[ood_test_idxs]
    test_data = data[ood_test_idxs]
    test_zipfile = {'data': test_imgs.astype(numpy.uint8),
                    'attributes': test_data.astype(numpy.uint8)}
    numpy.savez_compressed(
        outpath / f'mpi3d_{args.type}_ood_K{K}_seed{args.seed}.npz',
        **test_zipfile)
    del test_imgs
    del test_data
    del test_zipfile

    del imgs


if __name__ == "__main__":
    generate_dataset()

"""
re-not-double-descent | run_fig2.py | ishan kunam
-------------------------------------------------
re-run figure 2 of curth et al. (2023): random forests on mnist
fit one forest per leaf cap & score the first k trees for each ensemble size
test error is the squared error summed over the 10 classes
"""

import struct
from pathlib import Path

import numpy as np

# -------------------
# input configuration
# -------------------

ROOT = Path(__file__).resolve().parent.parent  # repository root
DATA_DIR = ROOT / "data" / "mnist"  # decompressed MNIST IDX files

LEAVES = [2, 5, 10, 20, 50, 75, 100, 500, 1000, 2000, None]  # P^leaf (None == no cap)
ENS = [1, 2, 5, 10, 20, 50, 100]  # P^ens

N_CLASSES = 10  # digits 0-9
N_TRAIN = 10000  # training images used per run
N_PER_CLASS = N_TRAIN // N_CLASSES  # images per class in the balanced subset
N_TREES = max(ENS)  # trees fitted per forest

# ---------------
# load mnist data
# ---------------


def read_idx(path: Path):
    # Read the raw bytes of the IDX file
    data = path.read_bytes()

    # Header: magic number (last byte == number of dimensions), then one size per dimension
    magic = struct.unpack(">I", data[:4])[0]
    ndim = magic & 0xFF
    dims = struct.unpack(">" + "I" * ndim, data[4 : 4 + 4 * ndim])

    # Remaining bytes are the array itself (unsigned 8-bit)
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load(data_dir: Path, subset: str = "first1000", subset_seed: int = 0):
    # Flatten each image to 784 features & scale pixels to [0, 1]
    Xtr = read_idx(data_dir / "train-images-idx3-ubyte").reshape(-1, 784)
    Xtr = Xtr.astype(np.float32) / 255.0
    ytr = read_idx(data_dir / "train-labels-idx1-ubyte").astype(int)
    Xte = read_idx(data_dir / "t10k-images-idx3-ubyte").reshape(-1, 784)
    Xte = Xte.astype(np.float32) / 255.0
    yte = read_idx(data_dir / "t10k-labels-idx1-ubyte").astype(int)

    # Pick the 10,000 training images
    if subset == "first1000":
        # First 1,000 images of every class, in dataset order
        idx = np.sort(
            np.concatenate(
                [np.where(ytr == c)[0][:N_PER_CLASS] for c in range(N_CLASSES)]
            )
        )
    else:
        # Uniform sample w/o replacement, ignoring class balance
        idx = np.sort(
            np.random.RandomState(subset_seed).choice(len(ytr), N_TRAIN, replace=False)
        )
    Xtr, ytr = Xtr[idx], ytr[idx]

    # One-hot labels => one regression target per class
    return Xtr, np.eye(N_CLASSES)[ytr], Xte, np.eye(N_CLASSES)[yte]

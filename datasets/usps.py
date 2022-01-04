import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from .utils import download_url, mkdir
from torch.utils.data import Dataset
import shutil
import urllib
import gzip
import pickle
import torchvision
import numpy as np
from PIL import Image

class USPS(Dataset):
    base_folder = 'images'
    url = "https://raw.githubusercontent.com/mingyuliutw/CoGAN/master/cogan_pytorch/data/uspssample/usps_28x28.pkl"
    def __init__(self, root, split='train', transforms=None, loader=default_loader, download=False, offset=0,labeloffset=0):
        self.root = root
        self.transforms = transforms
        self.loader = default_loader
        self.split = split
        self.offset = offset
        self.lboffset = labeloffset
        self.filename = "usps_28x28.pkl"

        if download:
            self.download()

        folder = ''
        self.folder=folder
        # self._load_metadata()

        self.train_data, self.train_labels = self._load_samples()

        if self.split == 'train':
            total_num_samples = self.train_labels.shape[0]
            indices = np.arange(total_num_samples)
            np.random.shuffle(indices)
            self.train_data = self.train_data[indices[0:self.dataset_size], ::]
            self.train_labels = self.train_labels[indices[0:self.dataset_size]]
        self.train_data *= 255.0
        self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC

        self.object_categories = list(range(10))
        print('USPS, Split: %s, Size: %d' % (self.split, self.__len__()))

    def _load_metadata(self):

        images = pd.read_csv(os.path.join(self.root, self.folder, 'image.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, self.folder, 'label.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, self.folder, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.split == 'train':
            self.data = self.data[self.data.is_training_img == 1]#img_id,filepath,target
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _load_samples(self):
        """Load sample images from dataset."""
        filename = os.path.join(self.root, self.filename)
        f = gzip.open(filename, "rb")
        data_set = pickle.load(f, encoding="bytes")
        f.close()
        if self.split == 'train':
            images = data_set[0][0]
            labels = data_set[0][1]
            self.dataset_size = labels.shape[0]
        else:
            images = data_set[1][0]
            labels = data_set[1][1]
            self.dataset_size = labels.shape[0]
        return images, labels
    def download(self):
        """Download dataset."""
        filename = os.path.join(self.root, self.filename)
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        if os.path.isfile(filename):
            return
        print("Download %s to %s" % (self.url, os.path.abspath(filename)))
        urllib.request.urlretrieve(self.url, filename)
        print("[DONE]")
        return

    # def __len__(self):
    #     return len(self.dataset_size)
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # sample = self.images.iloc[idx]
        # path = os.path.join(self.root,
        #                     self.base_folder, sample.filepath)

        target = self.train_labels[idx]-self.lboffset
        img = self.train_data[idx]
        # img = torchvision.transforms.ToPILImage(img)
        # img = Image.fromarray(np.uint8(img))

        if self.transforms is not None:
            # img = self.transforms(img.astype(np.uint8))
            img = self.transforms(img)
            # img = self.transforms(torchvision.transforms.ToPILImage(img))
        return img, target+self.offset

import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from .utils import download_url, mkdir
from torch.utils.data import Dataset
import shutil


class MNIST(Dataset):
    base_folder = 'images'
    def __init__(self, root, split='train', transforms=None, loader=default_loader, offset=0,labeloffset=0):
        self.root = root
        self.transforms = transforms
        self.loader = default_loader
        self.split = split
        self.offset = offset
        self.lboffset = labeloffset
        folder = ''
        # if self.lboffset == 0:
        #     folder = 'train_{}_{}'.format(self.lboffset, self.lboffset+9)
        # else:
        #     folder = 'train_{}_{}'.format(self.lboffset, self.lboffset+9)
        self.folder=folder
        self._load_metadata()
        self.object_categories = list(range(10))
        # if labeloffset == 0:
        #     self.object_categories = list(range(8))#识别7_9
        # else:
        #     self.object_categories = list(range(8))
        print('MNIST, Split: %s, Size: %d' % (self.split, self.__len__()))

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


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root,
                            self.folder, 'image', sample.filepath)
        target = sample.target-self.lboffset #7-9 -7
        img = self.loader(path)

        if self.transforms is not None:
            img = self.transforms(img)
        return img, target+self.offset
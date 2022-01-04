import os
from torch.utils import data
from torchvision import transforms
from datasets import CUB200, StanfordDogs,MNIST, USPS

def get_concat_dataloader(data_root, batch_size=64, download=False):
    transforms_train = transforms.Compose([
        transforms.Resize(size=224),
        transforms.RandomCrop(size=(224, 224)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(0.2),
        # transforms.RandomRotation(0.1),
        # transforms.RandomGrayscale(0.1),
        # transforms.ColorJitter(brightness=0.4, contrast=1.1, saturation=0.4, hue=1),
        # transforms.RandomApply([transforms1=transforms.RandomVerticalFlip(), transforms2=transforms.RandomRotation(), transforms3=transforms.RandomGrayscale()], p=0.5)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])
    transforms_val = transforms.Compose([
        transforms.Resize(size=224),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])

    transforms_train_usps = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize(size=224),
        transforms.Resize((224, 224)),
        # transforms.RandomCrop(size=(224, 224)),
        transforms.Grayscale(3),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    transforms_val_usps = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize(size=224),
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(size=(224, 224)),
        transforms.Grayscale(3),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    cub_root = os.path.join(data_root, 'cub200',)
    train_cub = CUB200(root=cub_root, split='train',
                        transforms=transforms_train,
                        download=download, offset=0)
    val_cub = CUB200(root=cub_root, split='test',
                        transforms=transforms_val,
                        download=False, offset=0)

    # mnist_root = os.path.join(data_root, 'mnist','train_0_9')
    # train_mnist = MNIST(root=mnist_root, split='train',
    #                           transforms=transforms_train,
    #                           offset=200)
    # val_mnist = MNIST(root=mnist_root, split='test',
    #                         transforms=transforms_val,
    #                          offset=200)  # add offset

    # usps_root = os.path.join(data_root, 'usps')
    # train_usps = USPS(root=usps_root, split='train',
    #                   transforms=transforms_train_usps,
    #                   download=download, offset=210)
    # val_usps = USPS(root=usps_root, split='test',
    #                 transforms=transforms_val_usps,
    #                 download=False, offset=210)  # add offset

    dogs_root = os.path.join(data_root, 'dogs')
    train_dogs = StanfordDogs(root=dogs_root, split='train',
                                transforms=transforms_train,
                                download=download, offset=200)
    val_dogs = StanfordDogs(root=dogs_root, split='test',
                            transforms=transforms_val,
                            download=False, offset=200) # add offset

    train_dst = data.ConcatDataset([train_cub,train_dogs])
    val_dst = data.ConcatDataset([val_cub, val_dogs])

    train_loader = data.DataLoader(
        train_dst, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(
        val_dst, batch_size=batch_size, drop_last=True, shuffle=False, num_workers=4)
    return train_loader, val_loader
import random
import os, sys
import numpy as np
from tqdm import tqdm
import argparse
import torch
from utils.stream_metrics import StreamClsMetrics, AverageMeter
from utils import mkdir_if_missing, Logger
from dataloader import get_concat_dataloader
from models.resnet import *
from models.densenet import *

_model_dict = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'densenet121': densenet121
}

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
flg = 0

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='./data')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model", type=str, default='resnet34')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--random_seed", type=int, default=1337)
    parser.add_argument("--download", action='store_true', default=False)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--cfl_lr", type=float, default=None)
    # parser.add_argument("--cfl_lr", type=float, default=6e-4)#loss下降，准确度不提升考虑是学习率太高。

    # parser.add_argument("--t1_ckpt", type=str, default='checkpoints/cub200_resnet18_best.pth')
    # parser.add_argument("--t2_ckpt", type=str, default='checkpoints/dogs_resnet34_best.pth')
    # parser.add_argument("--stu_ckpt", type=str, default='checkpoints/amal_resnet34_best_cub_mnist_927713.pth')#cub:0.723611,mnist:0.993049
    # parser.add_argument("--stu_ckpt", type=str, default='checkpoints/amal_resnet34_best_dogs_usps_8694.pth')#usps:0.991379, dogs: 0.843050
    # parser.add_argument("--stu_ckpt", type=str, default='checkpoints/amal_resnet34_best_partoverlap_891034.pth')#cub:0.753993,mnist:0.995663,usps:0.994379,dogs:0.869086
    # parser.add_argument("--stu_ckpt", type=str, default='checkpoints/amal_resnet34_epoch_4t_888550.pth')#cub:0.769583,mnist:0.993494,usps:0.876078,dogs:0.754548
    # parser.add_argument("--stu_ckpt", type=str, default='checkpoints/amal_resnet34_best_cubmnist_0.941014.pth')#cub:0.756701,mnist:0.993995
    # parser.add_argument("--stu_ckpt", type=str, default='checkpoints/amal_resnet34_best_4t_optimal.pth')#all:0.892772,cub:0.763993,mnist:0.993661,usps:0.907694,dogs:0.765191
    # parser.add_argument("--stu_ckpt", type=str, default='checkpoints/amal_resnet34_best_3t_optimal_gan_0.933178.pth')#cub:0.753160,mnist:0.995706,usps:0.8867457
    # parser.add_argument("--stu_ckpt", type=str, default='checkpoints/amal_resnet34_best_partoverlap_cfl.pth')#0.791443cub:0.738889,mnist:0.995107,0.199353,dogs:0.526586,

    parser.add_argument("--stu_ckpt", type=str, default='checkpoints/amal_resnet34_best_4torigin.pth')#cub:0.734549,mnist:0.993772,usps:0.894397,dogs:0.703941
    return parser

def validate(model, loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()
    with torch.no_grad():
        ct = 0
        for i, (images, labels) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach()  # .max(dim=1)[1].cpu().numpy()
            targets = labels  # .cpu().numpy()

            metrics.update(preds, targets)

        score = metrics.get_results()
    return score

def main():
    opts = get_parser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    # Set up random seed
    # mkdir_if_missing('checkpoints')
    # mkdir_if_missing('logs')
    # sys.stdout = Logger(os.path.join('logs', 'amal_%s.txt' % (opts.model)))  # Log存入对应名称文档
    print(opts)
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    cur_epoch = 0

    mkdir_if_missing('checkpoints')

    #  Set up dataloader
    train_loader, val_loader = get_concat_dataloader(data_root=opts.data_root, batch_size=opts.batch_size,
                                                     download=opts.download)

    stu = _model_dict[opts.model](pretrained=True, num_classes=340).to(device)
    metrics = StreamClsMetrics(340)

    print("validate on val set...")
    stu.load_state_dict(torch.load(opts.stu_ckpt)['model_state'])
    print("load ok")
    stu.eval()

    val_score = validate(model=stu,
                         loader=val_loader,
                         device=device,
                         metrics=metrics
                         )
    print(metrics.to_str(val_score))

    sys.stdout.flush()
if __name__ == '__main__':
    main()

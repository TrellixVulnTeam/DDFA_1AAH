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
    parser.add_argument("--gpu_id", type=str, default='3')
    parser.add_argument("--random_seed", type=int, default=1337)
    parser.add_argument("--download", action='store_true', default=False)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--cfl_lr", type=float, default=None)
    # parser.add_argument("--cfl_lr", type=float, default=6e-4)#loss下降，准确度不提升考虑是学习率太高。

    # parser.add_argument("--t1_ckpt", type=str, default='checkpoints/cub200_resnet18_best.pth')
    # parser.add_argument("--t2_ckpt", type=str, default='checkpoints/dogs_resnet34_best.pth')
    # parser.add_argument("--stu_ckpt", type=str, default='checkpoints/amal_resnet34_best_798828.pth')#cub
    # parser.add_argument("--stu_ckpt", type=str, default='checkpoints/amal_resnet34_best_cub_mnist_934476.pth')#cub:0.767396,mnist:0.994384
    #test CFL
    # parser.add_argument("--stu_ckpt", type=str, default='checkpoints/amal_resnet34_best_partoverlap_79141355.pth')#cub:0.734549,mnist:0.995162,usps:0.195043,dogs:0.530084

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

    stu = _model_dict[opts.model](pretrained=True, num_classes=320).to(device)
    metrics = StreamClsMetrics(320)

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

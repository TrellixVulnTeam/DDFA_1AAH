import random
import os, sys
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
from torch.utils import data

import torch
from PIL import Image
import matplotlib.pyplot as plt

from loss import SoftCELoss, CFLoss
from utils.stream_metrics import StreamClsMetrics, AverageMeter
from models.cfl import CFL_ConvBlock
from datasets import StanfordDogs, CUB200
from utils import mkdir_if_missing, Logger
from dataloader import get_concat_dataloader
from torchvision import transforms
from models.resnet import *
import torch.nn.functional as F
from models.densenet import *
from visualize import visualize_ct
import cv2

_model_dict = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'densenet121': densenet121
}

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# ct_t = [torch.zeros(128, 7, 7).to(device)] * 320 #这里不可以啊，因为是320维的，所以实在是太大了
flg = 0

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='./data')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model", type=str, default='resnet18')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--random_seed", type=int, default=1337)
    parser.add_argument("--download", action='store_true', default=False)
    parser.add_argument("--epochs", type=int, default=5)
    # parser.add_argument("--cfl_lr", type=float, default=None)
    parser.add_argument("--cfl_lr", type=float, default=None)#loss下降，准确度不提升考虑是学习率太高。

    parser.add_argument("--t_ckpt", type=list, default=[
                                                        'checkpoints/cub200_resnet34_81_90_best.pth',
                                                        'checkpoints/cub200_resnet34_91_100_best.pth',
                                                        'checkpoints/dogs_resnet34_70_79_best.pth',
                                                        'checkpoints/dogs_resnet34_100_109_best.pth',
                                                        'checkpoints/mnist_resnet18_0_3_best.pth',
                                                        'checkpoints/mnist_resnet18_4_6_best.pth',
                                                        'checkpoints/mnist_resnet18_7_9_best.pth',
                                                        ])
    # parser.add_argument("--t2_ckpt", type=str, default='checkpoints/cub200_resnet34_91_100_best.pth')
    # parser.add_argument("--t3_ckpt", type=str, default='checkpoints/dogs_resnet34_70_79_best.pth')
    # parser.add_argument("--t4_ckpt", type=str, default='checkpoints/dogs_resnet34_100_109_best.pth')
    # parser.add_argument("--t5_ckpt", type=str, default='checkpoints/mnist_resnet18_0_3_best.pth')
    # parser.add_argument("--t6_ckpt", type=str, default='checkpoints/mnist_resnet18_4_6_best.pth')
    # parser.add_argument("--t7_ckpt", type=str, default='checkpoints/mnist_resnet18_7_9_best.pth')

    return parser





def getcenterloss(ht_cluster, ct_t, device):
    loss_center_t = 0
    for label in ht_cluster.keys():
        #            for j in range(len(ht_cluster[item])):
        ht_cluster_ = torch.stack(ht_cluster[label]).view(len(ht_cluster[label]), -1)
        ct_t_ = ct_t[label].repeat(len(ht_cluster[label]), 1, 1, 1).view(len(ht_cluster[label]), -1)
        temp = torch.sum(torch.norm(ht_cluster_ - ct_t_, dim=1)).to(device).item()#矩阵范数这里再看一下对不对
        loss_center_t += temp / len(ht_cluster.keys())

    return loss_center_t


def getccloss_s(ct_s, v, device):
    loss_center_s = 0
    for i in range(len(ct_s) - 1):
        for j in range(i + 1, len(ct_s)):
            temp = torch.norm(ct_s[i] - ct_s[j]).to(device).item()
            loss_center_s += max(0, (v - temp)) / len(ct_s)

    return loss_center_s
def getccloss_ss(ct_s, v, device):
    loss_center_s = 0
    for i in range(len(ct_s) - 1):
        for j in range(i + 1, len(ct_s)):
            c_k1 = ct_s[list(ct_s.keys())[i]]
            c_k2 = ct_s[list(ct_s.keys())[j]]
            temp = torch.norm(c_k1 - c_k2).to(device).item()
            loss_center_ss += max(0, (v - temp)) / len(ct_s)
    return loss_center_ss


def amal(cur_epoch, criterion_ce, criterion_cf, model, cfl_blk, teachers, optim, train_loader, device, scheduler=None,
         print_interval=10):
    """Train and return epoch loss"""
#    t1, t2, t3, t4, t5, t6, t7 = teachers

    if scheduler is not None:
        scheduler.step()

    print("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))
    avgmeter = AverageMeter()
    is_densenet = isinstance(model, DenseNet)
    for cur_step, (images, labels) in enumerate(train_loader):

        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)  #
        # get soft-target logits
        optim.zero_grad()
        ft = []
        with torch.no_grad():
            t_out = []
            for t in teachers:
                t_out.append(t(images))
            t_outs = torch.cat(t_out, dim=1)
            for t in teachers:
                fti = t.layer4.output
                ft.append(fti)
        # get student output
        s_outs = model(images)
        if is_densenet:
            fs = model.features.output
        else:
            fs = model.layer4.output

#        ft = [ft1, ft2, ft3, ft4, ft5, ft6, ft7]

        (hs, ht), (ft_, ft) = cfl_blk(fs, ft)##################
        ######################################## Center_loss #################################################################################################

        #
        loss_ce = criterion_ce(s_outs, t_outs) #+ 0.1 * loss_center_ss

        loss_cf = 10 * criterion_cf(hs, ht, ft_, ft, labels, 10,device)
        ###############################################################################################################################################################
        loss = loss_ce + loss_cf
        loss.backward()
        optim.step()

        avgmeter.update('loss', loss.item())
        avgmeter.update('interval loss', loss.item())
        avgmeter.update('ce loss', loss_ce.item())
        avgmeter.update('cf loss', loss_cf.item())
        root = '/media/data/wlanyu/cfl/CommonFeatureLearning-master/'
        # if (criterion_cf.times) % 500 == 20:
        #      visualize_ct(criterion_cf.ct_t_data, criterion_cf.ht_cluster,os.path.join(root,'visualize','epoch:'+str(cur_epoch)+'_batch:'+str(cur_step + 1)+'.png'))

        if (cur_step + 1) % print_interval == 0:
            interval_loss = avgmeter.get_results('interval loss')
            ce_loss = avgmeter.get_results('ce loss')
            cf_loss = avgmeter.get_results('cf loss')

            print("Epoch %d, Batch %d/%d, Loss=%f (ce=%f, cf=%s)" %
                  (cur_epoch, cur_step + 1, len(train_loader), interval_loss, ce_loss, cf_loss))
            avgmeter.reset('interval loss')
            avgmeter.reset('ce loss')
            avgmeter.reset('cf loss')
    return avgmeter.get_results('loss')


def validate(model, loader, device, metrics,save,save_root):
    """Do validation and return specified samples"""
    if not os.path.isdir(save_root):
        os.mkdir(save_root)
    metrics.reset()
    with torch.no_grad():
        ct = 0
        for i, (images, labels) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach()  # .max(dim=1)[1].cpu().numpy()
            targets = labels  # .cpu().numpy()

#################################可视化错误样本
            if save == True:
                unloader = transforms.ToPILImage()
                output_class = preds.max(dim=1)[1]
                true_class = targets
                wrong_index = np.argwhere((output_class != true_class).cpu().numpy() == True)[:,0]
                wrong_class_images = images[wrong_index]
                for index, image in zip(wrong_index,wrong_class_images):
                    save_path = os.path.join(save_root, '%s.jpg' % (str(ct)))
                    image_ = image.cpu().clone()
                    image_ = unloader(image_)
                    image_.save(save_path)
                    f = open(os.path.join(save_root, 'wrong_class_record.txt'), 'a')
                    f.write('%s.jpg' % (str(ct)) + ' '+'%s'%str(int(output_class[index]))+' '+'%s'%str(int(true_class[index]))+'\n')
                    ct+=1
                    f.close()

                #  mnist

                # output_class = preds.max(dim=1)[1]
                # true_class = targets
                # wrong_index = np.argwhere((output_class != true_class).cpu().numpy() == True)[:,0]
                # wrong_class_images = images[wrong_index].cpu().numpy()
                # for index, image in zip(wrong_index,wrong_class_images):
                #     save_path = os.path.join(save_root, '%s.jpg' % (str(ct)))
                #     r,g,b = image
                #     cv2_img = cv2.merge([b, g, r])
                #     cv2_img -= np.min(cv2_img[cv2_img > 0])
                #     cv2_img = cv2_img / np.max(cv2_img) * 255
                #     cv2_img = np.uint8(cv2_img)
                #     cv2.imwrite(save_path, cv2_img)
                #     f = open(os.path.join(save_root, 'wrong_class_record.txt'), 'a')
                #     f.write('%s.jpg' % (str(ct)) + ' '+'%s'%str(int(output_class[index]))+' '+'%s'%str(int(true_class[index]))+'\n')
                #     ct+=1
                #     f.close()


                # cub200
                output_class = preds.max(dim=1)[1]
                true_class = targets
                wrong_index = np.argwhere((output_class != true_class).cpu().numpy() == True)[:,0]
                wrong_class_images = images[wrong_index].cpu().numpy()
                for index, image in zip(wrong_index,wrong_class_images):
                    save_path = os.path.join(save_root, '%s.jpg' % (str(ct)))
                    r,g,b = image
                    mean=[0.485, 0.456, 0.406]
                    std=[0.229, 0.224, 0.225]
                    r,g,b = ((ch*std+mean)*255 for ch,std,mean in zip([r,g,b],std,mean))
                    cv2_img = cv2.merge([b, g, r])
                    # cv2_img -= np.min(cv2_img[cv2_img > 0])
                    # cv2_img = cv2_img / np.max(cv2_img) * 255
                    cv2_img = np.uint8(cv2_img)
                    cv2.imwrite(save_path, cv2_img)
                    f = open(os.path.join(save_root, 'wrong_class_record.txt'), 'a')
                    f.write('%s.jpg' % (str(ct)) + ' '+'%s'%str(int(output_class[index]))+' '+'%s'%str(int(true_class[index]))+'\n')
                    ct+=1
                    f.close()
#################################

            metrics.update(preds, targets)

        score = metrics.get_results()
    return score


def main():
    opts = get_parser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    # Set up random seed
    mkdir_if_missing('checkpoints')
    mkdir_if_missing('logs')
    sys.stdout = Logger(os.path.join('logs', 'amal_%s.txt' % (opts.model)))  # Log存入对应名称文档
    print(opts)
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    cur_epoch = 0
    best_score = 0.0

    mkdir_if_missing('checkpoints')
    latest_ckpt = 'checkpoints/amal_%s_latest.pth' % opts.model  # 存入最终结果
    best_ckpt = 'checkpoints/amal_%s_best.pth' % opts.model  # 存入最好结果

    #  Set up dataloader
    train_loader, val_loader = get_concat_dataloader(data_root=opts.data_root, batch_size=opts.batch_size,
                                                     download=opts.download)

    # pretrained teachers
    t1_model_name = opts.t_ckpt[0].split('/')[1].split('_')[1]
    t1 = _model_dict[t1_model_name](num_classes=10).to(device)  # cub200_resnet34_81_90
    t2_model_name = opts.t_ckpt[1].split('/')[1].split('_')[1]
    t2 = _model_dict[t2_model_name](num_classes=10).to(device)  # cub200_resnet34_91_100
    # t3_model_name = opts.t_ckpt[2].split('/')[1].split('_')[1]
    # t3 = _model_dict[t3_model_name](num_classes=10).to(device)  # dogs_resnet34_70_79
    # t4_model_name = opts.t_ckpt[3].split('/')[1].split('_')[1]
    # t4 = _model_dict[t4_model_name](num_classes=10).to(device)  # dogs_resnet34_100_109
    # t5_model_name = opts.t_ckpt[4].split('/')[1].split('_')[1]
    # t5 = _model_dict[t5_model_name](num_classes=4).to(device)  # mnist_resnet18_0_3
    # t6_model_name = opts.t_ckpt[5].split('/')[1].split('_')[1]
    # t6 = _model_dict[t6_model_name](num_classes=3).to(device)  # mnist_resnet18_4_6
    # t7_model_name = opts.t_ckpt[6].split('/')[1].split('_')[1]
    # t7 = _model_dict[t7_model_name](num_classes=3).to(device)  # mnist_resnet18_7_9
    # ts = [t1,t2,t3,t4,t5,t6,t7]
    ts = [t1,t2]
#    print("Loading pretrained teachers ...\nT1: %s, T2: %s, T3: %s, T2: %s, T2: %s, T2: %s, T2: %s" % (t1_model_name, t2_model_name))
    for i in range(len(ts)):
        t = ts[i]
        t.load_state_dict(torch.load(opts.t_ckpt[i])['model_state'])  # 加载参数 #训练mnist修改此处
        t.eval()
    print("Target student: %s" % opts.model)
    stu = _model_dict[opts.model](pretrained=True, num_classes=20).to(device)#num_class
    metrics = StreamClsMetrics(20)#num_class

    # Setup Common Feature Blocks
    t_feature_dim = []
    for t in ts:
        t_feature_dim.append(t.fc.in_features)

    is_densenet = True if 'densenet' in opts.model else False

    if is_densenet:
        stu_feature_dim = stu.classifier.in_features
    else:
        stu_feature_dim = stu.fc.in_features

    cfl_blk = CFL_ConvBlock(stu_feature_dim, t_feature_dim, 128).to(
        device)  ##################################

    def forward_hook(module, input, output):
        module.output = output  # keep feature maps

    for t in ts:
        t.layer4.register_forward_hook(forward_hook)

    if is_densenet:
        stu.features.register_forward_hook(forward_hook)
    else:
        stu.layer4.register_forward_hook(forward_hook)

    params_1x = []
    params_10x = []
    for name, param in stu.named_parameters():
        if 'fc' in name:
            params_10x.append(param)
        else:
            params_1x.append(param)

    cfl_lr = opts.lr * 10 if opts.cfl_lr is None else opts.cfl_lr
    optimizer = torch.optim.Adam([{'params': params_1x, 'lr': opts.lr},
                                  {'params': params_10x, 'lr': opts.lr * 10},
                                  {'params': cfl_blk.parameters(), 'lr': cfl_lr}],
                                 lr=opts.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=15, gamma=0.1)

    # Loss
    criterion_ce = SoftCELoss(T=1.0)
    criterion_cf = CFLoss(normalized=True)

    def save_ckpt(path):
        """ save current model
        """
        state = {
            "epoch": cur_epoch,
            "model_state": stu.state_dict(),
            "cfl_state": cfl_blk.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }
        torch.save(state, path)
        print("Model saved as %s" % path)


    print("Training ...")
    # ===== Train Loop =====#
    while cur_epoch < opts.epochs:
        stu.train()
        epoch_loss = amal(cur_epoch=cur_epoch,
                          criterion_ce=criterion_ce,
                          criterion_cf=criterion_cf,
                          model=stu,
                          cfl_blk=cfl_blk,
                          teachers=ts,
                          optim=optimizer,
                          train_loader=train_loader,
                          device=device,
                          scheduler=scheduler)
        print("End of Epoch %d/%d, Average Loss=%f" %
              (cur_epoch, opts.epochs, epoch_loss))

        # =====  Latest Checkpoints  =====
        save_ckpt(latest_ckpt)
        # =====  Validation  =====
        print("validate on val set...")
        stu.eval()
        save=False
        save_root= 'misclassification'
        if cur_epoch == opts.epochs-1:
            save=True
        val_score = validate(model=stu,
                             loader=val_loader,
                             device=device,
                             metrics=metrics,
                             save=save,
                             save_root=save_root)
        print(metrics.to_str(val_score))
        sys.stdout.flush()
        # =====  Save Best Model  =====
        if val_score['Overall Acc'] > best_score:  # save best model
            best_score = val_score['Overall Acc']
            save_ckpt(best_ckpt)
        cur_epoch += 1
        if cur_epoch % 10==0:
            ckpt_name = 'checkpoints/epoch/amal_{0}_epoch_{1}.pth'.format(opts.model, cur_epoch)
            save_ckpt(ckpt_name)
if __name__ == '__main__':
    main()

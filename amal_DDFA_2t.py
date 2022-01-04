import random
import os, sys
import numpy as np
from tqdm import tqdm
import argparse
# import cv2
import torch
import torch.nn as nn
from torch.utils import data

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
from models.cam import CAM
_model_dict = {
    # 'vgg16': vgg16,
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
    parser.add_argument("--model", type=str, default='resnet34')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--random_seed", type=int, default=1337)
    parser.add_argument("--download", action='store_true', default=False)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--cfl_lr", type=float, default=None)
    # parser.add_argument("--cfl_lr", type=float, default=6e-4)#loss下降，准确度不提升考虑是学习率太高。

    parser.add_argument("--t1_ckpt", type=str, default='checkpoints/cub200_resnet18_best.pth')
    # parser.add_argument("--t2_ckpt", type=str, default='checkpoints/mnist_resnet18_0_9_best_997164.pth')
    # parser.add_argument("--t1_ckpt", type=str, default='checkpoints/USPS_resnet18_0_9_best_993534.pth')
    parser.add_argument("--t2_ckpt", type=str, default='checkpoints/dogs_resnet34_best.pth')
    # parser.add_argument("--t1_ckpt", type=str, default='checkpoints/amal_resnet34_best_cub_mnist_934476.pth')
    # parser.add_argument("--t2_ckpt", type=str, default='checkpoints/amal_resnet34_best_dogs_usps_843271.pth')
    # parser.add_argument("--stu_ckpt", type=str, default='checkpoints/epoch8b4b2bp03beta07l6t507792/amal_resnet34_epoch_40.pth')

    return parser





def amal(cur_epoch, criterion_ce, criterion_cf, model, cfl_blk,cam, teachers, optim, train_loader, device, scheduler=None,
         print_interval=10):
    """Train and return epoch loss"""
    t1, t2 = teachers

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
        with torch.no_grad():
            t1_out = t1(images)
            t2_out = t2(images)
            t_outs = torch.cat((t1_out, t2_out), dim=1)#concat 两个类的作为标签。

            ft1 = t1.layer4.output
            ft2 = t2.layer4.output

        # get student output
        s_outs = model(images)
        if is_densenet:
            fs = model.features.output
        else:
            fs = model.layer4.output

        ft = [ft1, ft2]

        (hs, ht), (ft_, ft) = cfl_blk(fs, ft)
        #cross attention
        (ht, hs) = cam(ht, hs)
        ######################################## Center_loss #################################################################################################

        #
        loss_ce = criterion_ce(s_outs, t_outs) #+ 0.1 * loss_center_ss
        # loss_cf = 10 * criterion_cf(hs, ht, ft_, ft, labels, 10)
        loss_cf = 10 * criterion_cf(hs, ht, ft_, ft, labels, 10, device)
        ###############################################################################################################################################################
        loss = loss_ce + loss_cf
        loss.backward()
        optim.step()

        avgmeter.update('loss', loss.item())
        avgmeter.update('interval loss', loss.item())
        avgmeter.update('ce loss', loss_ce.item())
        avgmeter.update('cf loss', loss_cf.item())
        root = '/media/data/shuoying/CommonFeatureLearning-master/'
        # if (criterion_cf.times) % 800 == 100:
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


# def validate(model, loader, device, metrics):
#     """Do validation and return specified samples"""
#     metrics.reset()
#     with torch.no_grad():
#         for i, (images, labels) in tqdm(enumerate(loader)):
#             images = images.to(device, dtype=torch.float32)
#             labels = labels.to(device, dtype=torch.long)
#
#             outputs = model(images)
#
#             preds = outputs.detach()  # .max(dim=1)[1].cpu().numpy()
#             targets = labels  # .cpu().numpy()
#
#             metrics.update(preds, targets)
#         score = metrics.get_results()
#     return score

def validate(model, loader, device, metrics, save, save_root):
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

            #################################
            if save == True:
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

                # test
                # unloader = transforms.ToPILImage()
                # output_class = preds.max(dim=1)[1]
                # true_class = targets
                # wrong_index = np.argwhere((output_class != true_class).cpu().numpy() == True)[:,0]
                # wrong_class_images = images[wrong_index]
                # for index, image in zip(wrong_index,wrong_class_images):
                #     save_path = os.path.join(save_root, '%s.jpg' % (str(ct)))
                #     image_ = image.cpu().clone()
                #     image_ = unloader(image_)
                #     image_.save(save_path)
                #     f = open(os.path.join(save_root, 'wrong_class_record.txt'), 'a')
                #     f.write('%s.jpg' % (str(ct)) + ' '+'%s'%str(int(output_class[index]))+' '+'%s'%str(int(true_class[index]))+'\n')
                #     ct+=1
                #     f.close()

                # cub200
                output_class = preds.max(dim=1)[1]
                true_class = targets
                wrong_index = np.argwhere((output_class != true_class).cpu().numpy() == True)[:, 0]
                wrong_class_images = images[wrong_index].cpu().numpy()
                for index, image in zip(wrong_index, wrong_class_images):
                    save_path = os.path.join(save_root, '%s.jpg' % (str(ct)))
                    r, g, b = image
                    mean = [0.485, 0.456, 0.406]
                    std = [0.229, 0.224, 0.225]
                    r, g, b = ((ch * std + mean) * 255 for ch, std, mean in zip([r, g, b], std, mean))
                    cv2_img = cv2.merge([b, g, r])
                    # cv2_img -= np.min(cv2_img[cv2_img > 0])
                    # cv2_img = cv2_img / np.max(cv2_img) * 255
                    cv2_img = np.uint8(cv2_img)
                    cv2.imwrite(save_path, cv2_img)
                    f = open(os.path.join(save_root, 'wrong_class_record.txt'), 'a')
                    f.write('%s.jpg' % (str(ct)) + ' ' + '%s' % str(int(output_class[index])) + ' ' + '%s' % str(
                        int(true_class[index])) + '\n')
                    ct += 1
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
    sys.stdout = Logger(os.path.join('logs', 'amal_%s_%s.txt' % (opts.model,'eval')))  # Log存入对应名称文档
    print(opts)
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    cur_epoch = 0
    best_score = 0.0

    mkdir_if_missing('checkpoints')
    latest_ckpt = 'checkpoints/amal_%s_latest.pth' % opts.model  # 存入最终结果

    # latest_ckpt = 'checkpoints/epoch8b4b2bp03beta07l6t507792/amal_resnet34_epoch_40.pth'   # 存入最终结果
    best_ckpt = 'checkpoints/amal_%s_best.pth' % opts.model  # 存入最好结果

    #  Set up dataloader
    train_loader, val_loader = get_concat_dataloader(data_root=opts.data_root, batch_size=opts.batch_size,
                                                     download=opts.download)

    # pretrained teachers
    t1_model_name = opts.t1_ckpt.split('/')[1].split('_')[1]
    t1 = _model_dict[t1_model_name](num_classes=200).to(device)  # cub200
    t2_model_name = opts.t2_ckpt.split('/')[1].split('_')[1]
    t2 = _model_dict[t2_model_name](num_classes=120).to(device)  # dogs
    print("Loading pretrained teachers ...\nT1: %s, T2: %s" % (t1_model_name, t2_model_name))
    t1.load_state_dict(torch.load(opts.t1_ckpt)['model_state'])  # 加载参数
    t2.load_state_dict(torch.load(opts.t2_ckpt)['model_state'])
    t1.eval()
    t2.eval()

    print("Target student: %s" % opts.model)
    stu = _model_dict[opts.model](pretrained=True, num_classes=200 + 120).to(device)
    metrics = StreamClsMetrics(200 + 120)

    # Setup Common Feature Blocks
    t1_feature_dim = t1.fc.in_features
    t2_feature_dim = t2.fc.in_features

    is_densenet = True if 'densenet' in opts.model else False

    if is_densenet:
        stu_feature_dim = stu.classifier.in_features
    else:
        stu_feature_dim = stu.fc.in_features

    cfl_blk = CFL_ConvBlock(stu_feature_dim, [t1_feature_dim, t2_feature_dim], 128).to(
        device)  ##################################
    cam = CAM().to(device)

    def forward_hook(module, input, output):
        module.output = output  # keep feature maps

    t1.layer4.register_forward_hook(forward_hook)
    t2.layer4.register_forward_hook(forward_hook)

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
                                  {'params': cfl_blk.parameters(), 'lr': cfl_lr},
                                  {'params': cam.parameters(), 'lr':0.1}],
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
            "cam_state": cam.state_dict(),
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
                          cam=cam,
                          teachers=[t1, t2],
                          optim=optimizer,
                          train_loader=train_loader,
                          device=device,
                          scheduler=scheduler)
        print("End of Epoch %d/%d, Average Loss=%f" %
              (cur_epoch, opts.epochs, epoch_loss))

        # =====  Latest Checkpoints  =====
        # save_ckpt(latest_ckpt)
        # =====  Validation  =====
        print("validate on val set...")
        # t2.load_state_dict(torch.load(opts.stu_ckpt)['model_state'])
        # print("load ok")
        stu.eval()
        save = False
        save_root = 'misclassification'
        if cur_epoch == opts.epochs - 1:
            save = False
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
    print('best_score:',best_score)
if __name__ == '__main__':
    main()
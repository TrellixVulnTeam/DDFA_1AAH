import random
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os
import argparse
import sys
import os
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cfl import CFL_ConvBlock
from dataloader import get_concat_dataloader
from utils import mkdir_if_missing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.resnet import *
from models.densenet import densenet121
from matplotlib import colors as mcolors

_model_dict = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'densenet121': densenet121
}

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='./data')
    parser.add_argument("--gpu_id", type=str, default='4')
    parser.add_argument("--random_seed", type=int, default=1337)
    parser.add_argument("--download", action='store_true', default=False)
    parser.add_argument("--only_kd", action='store_true', default=False)
    parser.add_argument("--epochs", type=int, default=60)
    return parser

def get_samples(loader, classes, sample_num=10):
    class_sample_num = { c: 0 for c in classes }
    samples = []
    samples_lbl = []
    finished_class = 0
    for itr_cnt, (images, labels) in enumerate(loader):
        for img, lbl in zip(images, labels):
            lbl = int(lbl.numpy())
            if lbl in classes and class_sample_num[lbl]<sample_num:
                samples.append(img)
                samples_lbl.append(lbl)
                class_sample_num[lbl]+=1

                if class_sample_num[lbl]==sample_num:
                    finished_class+=1
                    if finished_class==len(classes):
                        return torch.stack( samples, dim=0 ), torch.from_numpy( np.array(samples_lbl) )
    return torch.stack( samples, dim=0 ), torch.from_numpy( np.array(samples_lbl) )

def main():
    opts = get_parser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Set up random seed
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    ckpt_dir = './checkpoints'
    # Set up dataloader
    train_loader, val_loader = get_concat_dataloader(data_root=opts.data_root, batch_size=64, download=opts.download)
    ckpt_list = os.listdir(os.path.join(ckpt_dir,'epochorigin'))
    for ckpt_name in ckpt_list:
        t_ckpt_path = []
        stu_ckpt = os.path.join(ckpt_dir, 'epochorigin',ckpt_name)

        ts_model_name = []
        for t_ckpt in t_ckpt_path:
            ts_model_name.append(t_ckpt.split('/')[2].split('_')[1])
        stu_model_name = stu_ckpt.split('/')[3].split('_')[1]

        num_classes_stu = 340
        stu = _model_dict[stu_model_name](num_classes=num_classes_stu).to(device)

        print("Loading student: %s"%(stu_model_name))
        stu.load_state_dict( torch.load(stu_ckpt, map_location=lambda storage, loc: storage)['model_state'] )

        is_densenet = True if 'densenet' in stu_model_name else False
        if is_densenet:
            stu_feature_dim = stu.classifier.in_features
        else:
            stu_feature_dim = stu.fc.in_features

        def forward_hook(module, input, output):
            module.output = output # keep feature maps

        if is_densenet:
            stu.features.register_forward_hook(forward_hook)
        else:
            stu.layer4.register_forward_hook(forward_hook)


        stu.eval()


        mkdir_if_missing('tsne_results')
        mkdir_if_missing('tsne_results/%s'%stu_model_name)

        marker_list = [ '^', 's','*','+' ]
        with torch.no_grad():

            for j in range(5):
                print("Split %d/10"%j)
                # class_list = np.arange(j*20, (j+1)*20)
                class_list = np.arange(j,340,5)


                colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
                # class_list = [j]
                cmap = matplotlib.cm.get_cmap("tab20").colors+matplotlib.cm.get_cmap("tab20b").colors+matplotlib.cm.get_cmap("tab20c").colors+matplotlib.cm.get_cmap("Accent").colors
                cmap = [(*item,1.0) for item in cmap]
                # cmap2 = matplotlib.cm.get_cmap('tab20b')
                # cmap3 = matplotlib.cm.get_cmap('tab20c')
                # cmap4 = matplotlib.cm.get_cmap('pastel1')
                # cmap = torch.stack(cmap1,cmap2,cmap3,cmap4)
                # colors =['black','gray','Silver','rosybrown','firebrick','red','darksalmon','sienna','sandybrown','bisque','tan','moccasin','gold','darkkhaki','olivedrab','chartreuse','palegreen','darkgreen','seagreen','lightseagreen','paleturquoise','darkcyan','darkturquoise','deepskyblue','aliceblue','slategray','royalblue','slategray','royalblue','navy','blue','mediumpurple','darkorchid','plum','m','mediumvioletred','palevioletred','grey','lightcoral','maroon','coral','peachpuff','darkorange','navajowhite','orange','darkgoldenrod','olive','yellowgreen','lawngreen','lightgreen','g','slateblue','darkciolet','fuchsia','darkviolet','violet','deeppink','crimson','indianred','salmon','darkred','peru','burlywood','y','aqua','pink','lime','thistle']
                # TODO: make it fast.
                images, labels = get_samples(val_loader, class_list, 50)
                sample_class_num = len(class_list)
                print("%d samples selected"%len(images))

                print("[Common Space]Extracting features ...")

                fs = []

                for img, lbl in tqdm(zip(images, labels)):
                    img = img.unsqueeze(0).to(device, dtype=torch.float)
                    lbl = lbl.unsqueeze(0).to(device, dtype=torch.long)

                    _ = stu(img)

                    if is_densenet:
                        _fs = stu.features.output
                    else:
                        _fs = stu.layer4.output



                    fs.append(_fs)



                fs = torch.cat(fs, dim=0)


                N, C, H, W = fs.shape
                features = [ fs.detach().view(N, -1) ]

                features = F.normalize( torch.cat( features, dim=0 ), p=2, dim=1 ).view(N, C, -1).mean(dim=2).cpu().numpy()#teacher_num+student_num
                # print("[Common Space] TSNE ...")
                tsne_res = TSNE(n_components=2, random_state=23333).fit_transform( features )
                # print("[Common Space] TSNE finished ...")

                print("features Ploting ... ")
                fig = plt.figure(1,figsize=(10,10))
                plt.axis('off')
                ax = fig.add_subplot(1, 1, 1)

                step_size = 1.0/sample_class_num
                labels = labels.detach().cpu().numpy()

                label_to_color = { class_list[i]: cmap[i%len(cmap)] for i in range(sample_class_num) }
                # label_to_color = {class_list[i]: colors[i] for i in range(sample_class_num)}
                sample_to_color = [ label_to_color[labels[i]] for i in range(len(labels))]
                ax.scatter(tsne_res[:N,0], tsne_res[:N, 1], c=sample_to_color, label = 'stu', marker="o", s = 30)

                ax.legend(fontsize="xx-large", markerscale=2)
                plt.show()
                plt.savefig('tsne_results/%s/epoch_%s_common_space_tsne_%d.svg'%(stu_model_name,stu_ckpt.split('_')[3].split('.')[0], j))
                plt.close()
if __name__ == '__main__':
    main()

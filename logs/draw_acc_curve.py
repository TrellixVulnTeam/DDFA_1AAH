import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import os, sys

# amal_log_paths = [ 'amal_resnet34.txt', 'amal_resnet50.txt', 'amal_densenet121.txt' ]
# kd_log_paths = [ 'kd_resnet34.txt', 'kd_resnet50.txt', 'kd_densenet121.txt' ]
# amal_log_paths = [ 'amal_resnet34.txt' ]

# amal_log_paths = [ 'amal_resnet18_cfl.txt']
# kd_log_paths = ['kd_resnet18.txt']
# DJP_log_paths = ['amal_resnet18_DJP.txt']
# DDKA_log_paths = ['amal_resnet18_ours.txt']
# CDAN_log_paths = ['amal_resnet18_CDAN.txt']


# amal_log_paths = [ 'amal_resnet34_7727.txt']
# kd_log_paths = [ 'kd_resnet34_50.txt', ]
# DJP_log_paths = [ 'amal_resnet34_7750.txt']
# DDKA_log_paths = [ 'amal_resnet34_7774.txt']
# CDAN_log_paths = [ 'amal_resnet34_CDAN.txt']

# amal_log_paths = [ 'amal_resnet50_CFL.txt']
# kd_log_paths = [ 'amal_resnet50_KD.txt', ]
# DJP_log_paths = [ 'amal_resnet50_DJP.txt']
# DDKA_log_paths = [ 'amal_resnet50_DDFA.txt']
# CDAN_log_paths = [ 'amal_resnet50_CDAN.txt']

amal_log_paths = [ 'amal_densenet121_CFL.txt']
kd_log_paths = [ 'kd_densenet121.txt', ]
DJP_log_paths = [ 'amal_densenet121_DJP.txt']
DDKA_log_paths = [ 'amal_densenet121_DDFA.txt']
CDAN_log_paths = [ 'amal_densenet121_CDAN.txt']

if __name__=='__main__':
    for amal_log_path, kd_log_path, DJP_log_path,DDKA_log_path,CDAN_log_path in zip(amal_log_paths, kd_log_paths,DJP_log_paths,DDKA_log_paths,CDAN_log_paths):
        # if not os.path.exists(amal_log_path) or not os.path.exists(kd_log_path) or not os.path.exists(DJP_log_path) or not os.path.exists(DDKA_log_path):
        #     continue
        amal_acc = []
        kd_acc = []
        DJP_acc = []
        DDKA_acc = []
        CDAN_acc =[]
        # model_name = amal_log_path.split('_')[1].split('.')[0]
        model_name = DDKA_log_path.split('_')[1].split('.')[0]

        with open(amal_log_path) as f:
            for line in f:
                if line[0]=='O': # Overall Acc
                    acc = float(line.strip('\n')[13:])
                    amal_acc.append(acc)
        with open(kd_log_path) as f:
            for line in f:
                if line[0]=='O': # Overall Acc
                    acc = float(line.strip('\n')[13:])
                    kd_acc.append(acc)
        with open(DJP_log_path) as f:
            for line in f:
                if line[0]=='O': # Overall Acc
                    acc = float(line.strip('\n')[13:])
                    DJP_acc.append(acc)

        with open(DDKA_log_path) as f:
            for line in f:
                if line[0]=='O': # Overall Acc
                    acc = float(line.strip('\n')[13:])
                    DDKA_acc.append(acc)

        with open(CDAN_log_path) as f:
            for line in f:
                if line[0]=='O': # Overall Acc
                    acc = float(line.strip('\n')[13:])
                    CDAN_acc.append(acc)
        
        plt.figure(1,figsize=(11,11))
        plt.xlim(0, 30)
        # plt.ylim(0.68, 0.78)
        # plt.ylim(0.695, 0.795)
        plt.ylim(0.685, 0.785)
        # plt.ylim(0.67, 0.745)
        plt.xlabel('epochs', fontsize=33)
        plt.ylabel('Accuracy', fontsize=33)
        plt.xticks(np.arange(0, 30, 6), fontproperties='Times New Roman', size=20)
        # plt.yticks(np.arange(0.67, 0.745, 0.015), fontproperties='Times New Roman', size=20)
        plt.yticks(np.arange(0.685, 0.785, 0.02), fontproperties='Times New Roman', size=20)
        plt.plot( list(range(len(amal_acc))), amal_acc, label="CFL", linewidth=5.0)
        plt.plot( list(range(len(kd_acc))), kd_acc, label="KD", linewidth=5.0)
        plt.plot( list(range(len(DJP_acc))), DJP_acc, label="DJP", linewidth=5.00)
        plt.plot( list(range(len(DDKA_acc))), DDKA_acc, label="DDFA", linewidth=5.0)
        plt.plot( list(range(len(CDAN_acc))), CDAN_acc, label="CDAN", linewidth=5.0)
        # plt.fill_between(range(50),
        #                  meanop - 1.96 * stdop,
        #                  meanop + 1.96 * stdop,
        #                  color='b',
        #                  alpha=0.2)

        plt.legend(loc='lower right', fontsize='xx-large' )
        plt.title(model_name, size=35)
        plt.savefig('acc-%s.svg'%model_name)
        plt.close()
        print('acc-%s.svg saved'%model_name)

    
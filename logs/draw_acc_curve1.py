import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os, sys

amal_log_path = 'teacher_mnist.txt'
# kd_log_paths = [ 'kd_resnet34_7727.txt', 'kd_resnet.txt', 'kd_densenet121.txt' ]


if __name__ == '__main__':
    amal_acc = []
    model_name = amal_log_path.split('_')[1].split('.')[0]
    with open(amal_log_path) as f:
        for line in f:
            if line[0] == 'O':  # Overall Acc
                acc = float(line.strip('\n')[13:])
                amal_acc.append(acc)
    plt.figure(1, figsize=(10, 10))
    plt.plot(list(range(len(amal_acc))), amal_acc, label="amal")
    plt.legend(loc='upper left', fontsize='xx-large')
    plt.title(model_name)
    plt.savefig('acc-%s.png' % model_name)
    plt.close()
    print('acc-%s.png saved' % model_name)


import numpy as np
from sklearn.manifold import TSNE
import torch
import matplotlib
matplotlib.use('Agg')
import pickle
import matplotlib.cm as cm
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt



def visualize_ct(ct,ht_cluster,filename):
    # ct_ = torch.stack(list(ct.values()))
    # N, C, H, W = ct_.shape
    # ct_ = ct_.view(N,-1)
    # ct_np = ct_.detach().cpu().numpy()
    # tsne = TSNE(n_components=2)
    # tsne.fit_transform(ct_np)
    cluster_dir = {}
    for key, value in ht_cluster.items():
        val_lst = list(value)
        # val_lst.append(ct[key])
        tmp = torch.stack(val_lst)
        N,C,H,W = tmp.shape
        tmp_ = tmp.view(N,-1)
        tmp_np = tmp_.detach().cpu().numpy()
        tsne = TSNE(n_components=2)
        tsne.fit_transform(tmp_np)
        cluster_dir[key] = tsne.embedding_

    plot_with_labels(cluster_dir, filename)

def plot_with_labels(low_dim_embs_dir, filename='tsne-300.svg'):
    plt.figure(figsize=(12, 12))  # in inches
    colors = cm.rainbow(np.linspace(0, 1, int(len(low_dim_embs_dir.keys()))))
    plt.ylim(-60,60)
    plt.xlim(-60,60)
    for low_dim_embs,color in zip(low_dim_embs_dir.values(),colors):
        x = low_dim_embs[:,0]
        y = low_dim_embs[:,1]
        plt.scatter(x, y,c=[color]*int(low_dim_embs.shape[0]))
    plt.savefig(filename)


# def plot_with_labels(low_dim_embs, filename='tsne-300.png'):
#     plt.figure(figsize=(12, 12))  # in inches
#     colors = cm.rainbow(np.linspace(0, 1, int(low_dim_embs.shape[0])))
#     for i,color in zip(range(int(low_dim_embs.shape[0])),colors):
#         x = low_dim_embs[i,0]
#         y = low_dim_embs[i,1]
#         plt.scatter(x, y,c=color)
#     plt.savefig(filename)




# def plot_with_labels(low_dim_embs_dir, filename='tsne-300.png'):
#     plt.figure(figsize=(12, 12))  # in inches
#     colors = cm.rainbow(np.linspace(0, 1, int(len(low_dim_embs_dir.keys()))))
#     for low_dim_embs,color in zip(low_dim_embs_dir.values(),colors):
#         for i in range(int(low_dim_embs.shape[0])):
#             x = low_dim_embs[i,0]
#             y = low_dim_embs[i,1]
#             plt.scatter(x, y,c=color)
#         plt.savefig(filename)







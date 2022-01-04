import torch
import torch.nn as nn
import torch.nn.functional as F
from .mmd import calc_mmd
import itertools
eta = 5
#recorder(loss_txishu:3,beta=0.6,lr_cfl=6e-4,p=0.2,acc=77.5729,times%500%200)
#recorder(loss_txishu: 3,beta=0.6,lr_cfl=6e-4,p=0.3,acc=77.1484,times%500%200)
#recorder(loss_txishu: 3,beta=0.6,lr_cfl=6e-4,p=0.2,acc=77.5251,times%500%100)
#recorder(loss_txishu: 3,beta=0.6,lr_cfl=6e-4,p=0.2,acc=77.4484,times%500%300)
#recorder(loss_txishu: 3,beta=0.6,lr_cfl=6e-4,p=0.2,acc=77.4344,times%200%100)
#recorder(loss_txishu: 3,beta=0.6,lr_cfl=6e-4,p=0.2,acc=77.4623,times%1000%200)
#recorder(loss_txishu: 5,beta=0.6,lr_cfl=6e-4,p=0.2,acc=77.4275,times%500%200)
#recorder(loss_txishu:3,beta=0.6,lr_cfl=5e-4,p=0.2,acc=77.3856,times%500%200)
#recorder(loss_txishu:3,beta=0.7,lr_cfl=6e-4,p=0.2,acc=77.4763,times%500%200)
#recorder(loss_txishu:3,beta=0.6,lr_cfl=6e-4,p=0.2,acc=76.6881,times%500%100tt5(v=5)%200t3)
#recorder(loss_txishu:3,beta=0.6,lr_cfl=6e-4,p=0.2,acc=77.07,times%500%100200tt5(v=5)%200t3)
#recorder(loss_txishu:3,beta=0.6,lr_cfl=6e-4,p=0.2,acc=,times%500%100200tt5(v=1.5)%200t3)
#recorder(loss_txishu:3,beta=0.6,lr_cfl=5e-4,p=0.2,acc=,times%500%100200tt5(v=1.5)%200t3)
#recorder(loss_txishu:3,beta=0.6,lr_cfl=5e-4,p=0.2,acc=77.4554,times%500%100200tt1(v=100)%200t3),loss降低了很多
#recorder(loss_txishu:3,beta=0.6,lr_cfl=5e-4,p=0.2,acc=77.2112,times%500%100250tt1(v=100)%250t3),loss降低了很多，但是还是得出现了acc不高。
#recorder(loss_txishu:3,beta=0.5,lr_cfl=5e-4,p=0.2,acc=77.2112,times%500%100250tt1(v=100)%250t3),loss降低了很多，但是还是得出现了acc不高。
#recorder(loss_txishu:3,beta=0.5,lr_cfl=5e-4,p=0.2,acc=,times%500%100250tt1(v=100)%250t3),loss降低了很多，但是还是得出现了acc不高。
class CFLoss(nn.Module):
    """ Common Feature Learning Loss
        CF Loss = MMD + beta * MSE
    """
    def __init__(self, sigmas=[0.001, 0.01, 0.05, 0.1, 0.2, 1, 2], normalized=True, beta=0.7):
        super(CFLoss, self).__init__()
        self.sigmas = sigmas
        self.normalized = normalized
        self.beta = beta
        self.ct_t = {}
        self.ct_t_data ={}
        self.ct_s = {}
        self.times = torch.tensor(0.)
        self.cir_times = torch.tensor(0.)
        self.ht_cluster = {}
    # def forward(self, hs, ht, ft_, ft, labels, v):
    #使用克罗内克积后的函数参数
    def forward(self, hs, ht, ft_, ft, labels, v, device):#hs,ht是common特征，ft_和ft是map back
        mmd_loss = 0#torch.tensor(0.0)
        mse_loss = 0#torch.tensor(0.0)
        # temp = ht+[hs]
        # ht_combinations = list(itertools.combinations(temp,2))


        # for ht_i in ht_combinations:
        #     # mmd_loss += calc_mmd(hs, ht_i, sigmas=self.sigmas, normalized=self.normalized)
        #     #使用克罗内克积的结果
        #     mmd_loss += calc_mmd(ht_i[0], ht_i[1], sigmas=self.sigmas, labels=labels, device=device, normalized=self.normalized)
        for ht_i in ht:
            for hs_i in hs:
                mmd_loss += calc_mmd(hs_i, ht_i, sigmas=self.sigmas, labels=labels,device=device,normalized=self.normalized)

        ###dual--idea##############################
        ht_combinations = list(itertools.combinations(ht,2))
        for ht_i in ht_combinations:
            mmd_loss += calc_mmd(ht_i[0], ht_i[1], sigmas=self.sigmas, labels=labels, device=device,normalized=self.normalized)
        for i in range(len(ft_)):
            mse_loss += F.mse_loss(ft_[i], ft[i])


        self.times += 1

        ht_cluster,ct_tmp = self.getcenter(ht, labels)

        if self.times % 800 == 0:
            self.ct_t_data = {}
            self.ct_t = {}
            self.ht_cluster = {}
        for key in ht_cluster.keys():
            if key not in self.ht_cluster.keys():
                self.ht_cluster[key] = []
            self.ht_cluster[key] = self.ht_cluster[key] + [elm.data for elm in ht_cluster[key]]

        loss_center_tt=0
        loss_center_tt_global=0
        loss_center_t =0
        # print('len_ct_t:',len(self.ct_t.keys()))
        if self.times >200 and self.times % 200 > 100:
              # loss_center_tt = getccloss_tt(ct_tmp, 10.0)
            loss_center_tt_global = getccloss_tt_global(ct_tmp,self.ct_t_data,50.0)
            # loss_center_t = getcenterloss(ht_cluster, ct_tmp)
            # print('loss_t:', loss_center_t)
            # if 100 <= self.times % 500 <= 250:
            #     loss_center_tt_global = getccloss_tt_global(ct_tmp, self.ct_t_data, 100)
            loss_center_t = getcenterloss(ht_cluster, ct_tmp)
            # print('loss_tt_global:', loss_center_tt_global)
            # print('loss_t:', loss_center_t)
        return mmd_loss + self.beta * mse_loss + 50 * loss_center_t + loss_center_tt_global


    def getcenter(self, commonfeature, labels):
        # times=torch.tensor(0.)
        h_cluster = {}
        # 实验一下不迭代的速度
        for i in range(len(labels)):  # batchsize
            label_int = int(labels[i])
            ct_tmp = {}# 若是不累加就不清空
            if label_int not in h_cluster.keys():
                h_cluster[label_int] = []
            if len(commonfeature)==2:
                if label_int not in self.ct_t_data.keys():
                    self.ct_t[label_int] = 0
                    self.ct_t_data[label_int]= 0
                    ct_tmp[label_int] = 0

            for j in range(len(commonfeature)):  # 判断是两个老师还是一个学生
                h_cluster[label_int].append(commonfeature[j][i])
        for label in h_cluster.keys():
            tmp = torch.stack(h_cluster[label])
            ct_new = tmp.mean(0)
            if self.times<=400:
                p= 1/self.times
            else:
                # p=0.3* torch.sin(0.01256 * self.times % 400)
                p=0.3
            # p = 0.2
            if len(commonfeature) == 2:
                #以下是具有累加效应的部分有backward 的类中心.
                self.ct_t[label] = (1-p) * self.ct_t_data[label] + p * ct_new
                #以下是每个batch下的含有backward的类中心.
                ct_tmp[label] = (1-p) * self.ct_t_data[label] + p * ct_new
                #以下是具有累加效应的类中心数值，没有backward.
                self.ct_t_data[label] = (1-p) * self.ct_t_data[label] + p * ct_new.data


        return h_cluster,ct_tmp

class SoftCELoss(nn.Module):
    """ KD Loss Function (CrossEntroy for soft targets)
    """
    def __init__(self, T=1.0, alpha=1.0):
        super(SoftCELoss, self).__init__()
        self.T = T
        self.alpha = alpha

    def forward(self, logits, targets, hard_targets=None):
        ce_loss = soft_cross_entropy(logits, targets, T=self.T)
        if hard_targets is not None and self.alpha != 0.0:
            ce_loss += self.alpha*F.cross_entropy(logits, hard_targets)
        return ce_loss

def soft_cross_entropy(logits, target, T=1.0, size_average=True, target_is_prob=False):
    """ Cross Entropy for soft targets

    **Parameters:**
        - **logits** (Tensor): logits score (e.g. outputs of fc layer)
        - **targets** (Tensor): logits of soft targets
        - **T** (float): temperature　of distill
        - **size_average**: average the outputs
        - **target_is_prob**: set True if target is already a probability.
    """
    if target_is_prob:
        p_target = target
    else:
        p_target = F.softmax(target/T, dim=1)

    logp_pred = F.log_softmax(logits/T, dim=1)
    # F.kl_div(logp_pred, p_target, reduction='batchmean')*T*T
    ce = torch.sum(-p_target * logp_pred, dim=1)
    if size_average:
        return ce.mean() * T * T
    else:
        return ce * T * T

def getcenterloss(ht_cluster, ct_t):
    loss_center_t = 0
    for label in ht_cluster.keys():
        #            for j in range(len(ht_cluster[item])):
        ht_cluster_ = torch.stack(ht_cluster[label]).view(len(ht_cluster[label]), -1)
        ct_t_ = ct_t[label].repeat(len(ht_cluster[label]), 1, 1, 1).view(len(ht_cluster[label]), -1)
        ht_cluster_ = F.normalize(ht_cluster_, p=2, dim=1)
        ct_t_ = F.normalize(ct_t_, p=2, dim=1)
        temp = torch.sum(torch.diag(torch.mm((ht_cluster_ - ct_t_), (ht_cluster_ - ct_t_).t())))#矩阵范数这里再看一下对不对
        loss_center_t += temp / len(ht_cluster.keys())

    return loss_center_t


def getccloss_tt(ct_t, v):
    ct = torch.stack(list(ct_t.values()))
    N, C, H, W = ct.shape
    ct = ct.view(N, -1)
    ct = F.normalize(ct, p=2, dim=1)
    xx = torch.mm(ct, ct.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    XX2 =rx.t() + rx - 2*xx
    print("XX2_max_min:", torch.max(torch.sqrt(torch.relu(XX2))), torch.min(torch.sqrt(torch.relu(XX2))))
    # XX2 = XX2_temp if torch.isnan(v-XX2_temp)==False and (v-XX2_temp)<0 else torch.tensor(0.)
    # v-XX2_new= torch.where(torch.isnan(v-XX2), torch.full_like(XX2, 0), 0)
    ct_count = len(ct_t)*(len(ct_t)-1)/2
    loss_center_tt = torch.sum(torch.relu(v-torch.sqrt(torch.relu(XX2)))/(ct_count*2))

    # print("v-XX2:",torch.max(v-XX2))
    return loss_center_tt

def getccloss_tt_global(ct_temp, ct_t, v):
    ct = torch.stack(list(ct_t.values()))
    N, C, H, W = ct.shape

    ct_b=torch.stack(list(ct_temp.values()))

    N_b, C_b, H_b, W_b = ct_b.shape

    ct_b = ct_b.view(N_b, -1)
    ct = ct.view(N, -1)
    ct_b = F.normalize(ct_b, p=2, dim=1)
    ct = F.normalize(ct, p=2, dim=1)

    xx = torch.mm(ct_b,ct_b.t())
    xy = torch.mm(ct_b, ct.t())
    yy = torch.mm(ct, ct.t())
    yy2 = yy.diag().unsqueeze(0).expand(len(ct_temp.keys()), len(ct_t.keys()))
    xx2 = xx.diag().unsqueeze(0).expand(len(ct_t.keys()), len(ct_temp.keys()))

    xy2 = xx2.t() + yy2 - 2*xy

    for key in ct_temp.keys():
        xy2[list(ct_temp.keys()).index(key)][list(ct_t.keys()).index(key)]= torch.tensor(0.)
        # print("keykey:",xy2[list(ct_temp.keys()).index(key)][list(ct_t.keys()).index(key)])


    ct_count = len(ct_b)*(len(ct_t)-1)
    loss_center_tt_global = torch.sum(torch.relu(v-torch.sqrt(torch.relu(xy2)))/ct_count)

    return loss_center_tt_global


def getccloss_ss(ct_s, v):
#    loss_center_tt = 0
    ct = torch.stack(list(ct_s.values()))
    N, C, H, W = ct.shape
    ct = ct.view(N, -1)
    ct = F.normalize(ct, p=2, dim=1)
    xx = torch.mm(ct, ct.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    XX2 = rx.t() + rx - 2*xx
    ct_count = len(ct_s)*(len(ct_s)-1)/2
    loss_center_ss = torch.sum(torch.relu(v-XX2)/(ct_count*2)).item()
    return loss_center_ss


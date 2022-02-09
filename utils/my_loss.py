import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb
from easydl import *

def Entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-7)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1
    
def Intance_weight(inputs,coeff):
    softmax_inputs = torch.nn.Softmax(dim=1)(inputs)
    entropy = Entropy(softmax_inputs)
    entropy.register_hook(grl_hook(coeff))
    w = 1.0 + torch.exp(-entropy)
    return w

def mse_loss(cluster_out, classifier_out,n_share):
    mse = nn.MSELoss(reduction='none').cuda()
    extend_cluster_out = torch.zeros_like(classifier_out)
    extend_cluster_out[:, 0:n_share] = cluster_out
    loss = mse(extend_cluster_out, classifier_out)
    loss = torch.mean(loss, dim=1)
    mloss = (loss).mean()
    return mloss
def cond_loss(cluster_source, cluster_target, switch_outputs, argu_cluster_source, argu_cluster_target, argu_switch_outputs, n_share, criterion):
    softmax_cluster_target = torch.nn.Softmax(dim=1)(cluster_target)
    cluster_label = torch.max(softmax_cluster_target,1)[1]
    softmax_switch_outputs = torch.nn.Softmax(dim=1)(switch_outputs)
    switch_label = torch.max(softmax_switch_outputs,1)[1]
    softmax_cluster_source = torch.nn.Softmax(dim=1)(cluster_source)
    cluster_source_label = torch.max(softmax_cluster_target,1)[1]

    argu_softmax_cluster_target = torch.nn.Softmax(dim=1)(argu_cluster_target)
    argu_cluster_label = torch.max(argu_softmax_cluster_target,1)[1]
    argu_softmax_switch_outputs = torch.nn.Softmax(dim=1)(argu_switch_outputs)
    argu_switch_label = torch.max(argu_softmax_switch_outputs,1)[1]
    argu_softmax_cluster_source = torch.nn.Softmax(dim=1)(argu_cluster_source)
    argu_cluster_source_label = torch.max(argu_softmax_cluster_target,1)[1]

    cond_targert_feature = torch.zeros_like(switch_outputs)
    cond_targert_feature[:, 0:n_share] = softmax_cluster_target

    argu_cond_targert_feature = torch.zeros_like(argu_switch_outputs)
    argu_cond_targert_feature[:, 0:n_share] = argu_softmax_cluster_target

    ts_lable = torch.cat([cluster_label, switch_label])                        #condition alligment
    ts_feature = torch.cat([cond_targert_feature, softmax_switch_outputs], dim = 0)
    nts_feature = F.normalize(ts_feature, dim=0)
    nt_feature = F.normalize(cond_targert_feature, dim=0)
    argu_ts_feature = torch.cat([cond_targert_feature, argu_softmax_switch_outputs], dim = 0)
    argu_nts_feature = F.normalize(argu_ts_feature, dim=0)
    argu_nt_feature = F.normalize(argu_cond_targert_feature, dim=0)
    #
    total_ts_feature = torch.cat([nts_feature.unsqueeze(1), argu_nts_feature.unsqueeze(1)], dim=1)
    condition_loss = criterion(total_ts_feature, ts_lable)

    # condition_loss = 0.0

    total_t_feature = torch.cat([nt_feature.unsqueeze(1), argu_nt_feature.unsqueeze(1)], dim=1)
    #total_s_feature = torch.cat([softmax_cluster_source.unsqueeze(1), argu_softmax_cluster_source.unsqueeze(1)], dim=1)
    cluster_loss = criterion(total_t_feature, cluster_label)#+criterion(total_s_feature, cluster_source_label)
    return condition_loss, cluster_loss

# def cond_loss(cluster_target, argu_cluster_target, criterion):
#     softmax_cluster_target = torch.nn.Softmax(dim=1)(cluster_target)
#     cluster_label = torch.max(softmax_cluster_target,1)[1]
#     nts_cluster_target = F.normalize(cluster_target, dim=0)
#     nt_argu_cluster_target = F.normalize(argu_cluster_target, dim=0)
#     nts_feature = torch.cat([nts_cluster_target.unsqueeze(1), nt_argu_cluster_target.unsqueeze(1)], dim=1)
#     #print('nts_feature',nts_feature)
#     cluster_loss = criterion(nts_feature, cluster_label)
#     return cluster_loss

def conloss(inputs, argu_inputs, labels, criterion):
    nswitch_outputs = F.normalize(inputs, dim=0)
    nargu_switch_outputs = F.normalize(argu_inputs, dim=0)
    total_switch_outputs = torch.cat([nswitch_outputs.unsqueeze(1), nargu_switch_outputs.unsqueeze(1)], dim=1)
    loss = criterion(total_switch_outputs, labels)
    return loss

def DANN(features, ad_net, entropy=None, coeff=None, cls_weight=None, len_share=0):
    ad_out = ad_net.forward(features)
    train_bs = (ad_out.size(0) - len_share) // 2
    dc_target = torch.from_numpy(np.array([[1]] * train_bs + [[0]] * (train_bs + len_share))).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0 + torch.exp(-entropy)
    else:
        entropy = torch.ones(ad_out.size(0)).cuda()

    source_mask = torch.ones_like(entropy)
    source_mask[train_bs : 2 * train_bs] = 0
    source_weight = entropy * source_mask
    source_weight = source_weight * cls_weight

    target_mask = torch.ones_like(entropy)
    target_mask[0 : train_bs] = 0
    target_mask[2 * train_bs::] = 0
    target_weight = entropy * target_mask
    target_weight = target_weight * cls_weight

    weight = (1.0 + len_share / train_bs) * source_weight / (torch.sum(source_weight).detach().item()) + \
            target_weight / (1e-8 + torch.sum(target_weight).detach().item())
        
    weight = weight.view(-1, 1)
    adv_loss = nn.BCELoss(reduction='none')(ad_out, dc_target)
    return torch.sum(weight * adv_loss) / (1e-8 + torch.sum(weight).item())

def marginloss(yHat, y, classes=65, alpha=1, weight=None):
    batch_size = len(y)
    classes = classes
    yHat = F.softmax(yHat, dim=1)
    Yg = torch.gather(yHat, 1, torch.unsqueeze(y, 1))#.detach()
    Yg_ = (1 - Yg) + 1e-7  # avoiding numerical issues (first)
    Px = yHat / Yg_.view(len(yHat), 1)
    Px_log = torch.log(Px + 1e-10)  # avoiding numerical issues (second)
    y_zerohot = torch.ones(batch_size, classes).scatter_(
        1, y.view(batch_size, 1).data.cpu(), 0)

    output = Px * Px_log * y_zerohot.cuda()
    loss = torch.sum(output, dim=1)/ np.log(classes - 1)
    Yg_ = Yg_ ** alpha
    if weight is not None:
        weight *= (Yg_.view(len(yHat), )/ Yg_.sum())
    else:
        weight = (Yg_.view(len(yHat), )/ Yg_.sum())

    weight = weight.detach()
    loss = torch.sum(weight * loss) / torch.sum(weight)

    return loss
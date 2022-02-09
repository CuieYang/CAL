import argparse
import os
import os.path as osp
import sys
sys.path.append(".")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import random
import pdb
import math
import json
from torchvision import transforms

import network.network as network
import utils.loss as loss
import utils.lr_schedule as lr_schedule
import dataset.preprocess as prep
from dataset.dataloader import ImageList, ImageList_idx
from distutils.version import LooseVersion
from scipy.spatial.distance import cdist
import utils.my_loss as my_loss
import torch.nn.functional as F
from utils.suplosses import SupConLoss
import data_list



class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, plain_transform, argu_transform):
        self.transform = plain_transform
        self.argu_transform = argu_transform

    def __call__(self, x):
        return [self.transform(x), self.argu_transform(x)]


def image_train(resize_size=256, crop_size=224):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


plain_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

argu_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def image_test(resize_size=256, crop_size=224):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def image_classification(loader, model,t_num):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader["test"])
        for i in range(len(loader['test'])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            fc1_s, _, outputs = model.forward(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    [max_val, indic] = torch.max(all_output, 1)
    avg_val = torch.mean(max_val)
    std = torch.std(max_val)
    median = torch.median(max_val)
    index = torch.ge(max_val, avg_val)
    index = torch.nonzero(index == True)
    index = torch.squeeze(index, dim=1)
    sect_output = all_output[index, :]
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    accuracyst = torch.sum(torch.squeeze(predict[index]).float() == all_label[index]).item() / float(all_label[index].size()[0] + 1e-8)
    mean_ent = torch.mean(my_loss.Entropy(torch.nn.Softmax(dim=1)(all_output))).cpu().data.item()
    shist_tar = torch.nn.Softmax(dim=1)(sect_output).sum(dim=0)
    shist_tar = shist_tar / shist_tar.sum()
    [_, sort_index] = torch.sort(shist_tar, descending=True)

    hist_tar = torch.nn.Softmax(dim=1)(all_output).sum(dim=0)
    hist_tar = hist_tar / hist_tar.sum()

    return accuracy, accuracyst, hist_tar, mean_ent, avg_val#+0.1*t_num*std

##test the model

def train_uda(config, args):

    ## set pre-process
    train_bs, test_bs = args.batch_size, args.batch_size * 2
    share_num = args.share_num

    dsets = {}
    dsets["source"] = data_list.ImageList(open(args.s_dset_path).readlines(), transform=TwoCropTransform(plain_transform, argu_transform))
    dsets["target"] = data_list.ImageList(open(args.t_dset_path).readlines(), transform=TwoCropTransform(plain_transform, argu_transform))
    dsets["test"] = data_list.ImageList(open(args.t_dset_path).readlines(), transform=image_test())

    dset_loaders = {}
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=True, num_workers=0, drop_last=True)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=0, drop_last=True)
    dset_loaders["test"]  = DataLoader(dsets["test"], batch_size=test_bs, shuffle=False, num_workers=0)

    ## set base network
    class_num = config["network"]["params"]["class_num"]
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()
    input_num = base_network.output_num()
    bottleneck_dim = 256

    ## add additional network for some methods
    ad_net = network.AdversarialNetwork(input_num, 1024)
    ad_net = ad_net.cuda()

    cluster_net = network.cluster(bottleneck_dim, share_num)
    cluster_net = cluster_net.cuda()

    switch_net = network.switch(bottleneck_dim, class_num)
    switch_net = switch_net.cuda()

    criterion = SupConLoss(temperature=args.temp)
    criterion = criterion.cuda()

    ## set optimizer
    parameter_list = base_network.get_parameters() + ad_net.get_parameters() + cluster_net.get_parameters()
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, \
                                         **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    # multi gpu
    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        ad_net = nn.DataParallel(ad_net, device_ids=[int(i) for i, k in enumerate(gpus)])
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i, k in enumerate(gpus)])

    ## train
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    max_len = 2*(len_train_source + len_train_target)

    class_weight = torch.from_numpy(np.array([1.0] * class_num))
    class_weight = class_weight / class_weight.sum()
    class_weight = class_weight.cuda()

    avg_val = 2.0
    t_num = 0

    varres={}
    varres["Test_acc"]=[]
    varres["Best_Test_acc"]=[]

    for i in range(config["num_iterations"]):
        # test
        if (i % max_len == 0 and i > 0) or (i == args.max_iterations):

            base_network.train(False)
            temp_acc, accuracyst, class_weight, mean_ent, avg_val = image_classification(dset_loaders, base_network,t_num)
            class_weight = class_weight.cuda().detach()
            log_str: str = "iter: {:05d}, precision: {:.5f} , precisionst: {:.5f}".format(i, temp_acc, accuracyst)
            print(log_str)
            t_num = t_num+1
            varres["Test_acc"].append(temp_acc)
            print('acc_test', temp_acc)
            with open("result.txt", 'w') as fw:
                fw.write(json.dumps(varres) + "\n")

        # save model
        if i % config["snapshot_interval"] == 0 and i:
            torch.save(base_network.state_dict(), osp.join(config["output_path"], "iter_{:05d}_model.pth.tar".format(i)))

        ## train one iter
        base_network.train(True)
        ad_net.train(True)
        cluster_net.train(True)

        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()

        # dataloader
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])

        im_source, label_source = iter_source.next()
        im_target, label_target = iter_target.next()

        label_source = label_source.cuda()

        im_source, argu_im_source = im_source[0].cuda(), im_source[1].cuda()
        im_target, argu_im_target = im_target[0].cuda(), im_target[1].cuda()

        # network
        feature_source, fea_bs, fc_s = base_network(im_source)
        feature_target, fea_bt, fc_t = base_network(im_target)
        argu_feature_source, argu_fea_bs, argu_fc_s = base_network(argu_im_source)
        argu_feature_target, argu_fea_bt, argu_fc_t = base_network(argu_im_target)

        [prec_fct, indic] = torch.max(fc_t, 1)
        indext = torch.ge(prec_fct, avg_val)
        indext = torch.nonzero(indext == True)
        indext = torch.squeeze(indext, dim=1)

        if class_weight is not None and args.weight_cls and class_weight[label_source].sum() == 0:
            continue

        cluster_target = cluster_net(fea_bt)
        cluster_source = cluster_net(fea_bs)
        switch_outputs = switch_net(fea_bs)

        argu_cluster_target = cluster_net(argu_fea_bt)
        argu_cluster_source = cluster_net(argu_fea_bs)
        argu_switch_outputs = switch_net(argu_fea_bs)

        nfc_t = F.normalize(fc_t, dim=0)
        nargu_fc_t = F.normalize(argu_fc_t, dim=0)
        nfc_s = F.normalize(fc_s, dim=0)
        nargu_fc_s = F.normalize(argu_fc_s, dim=0)
        total_feature_target = torch.cat([nfc_t.unsqueeze(1), nargu_fc_t.unsqueeze(1)], dim=1)

        if t_num > 0:
            sect_fct = nfc_t[indext, :]
            ect_argufct = nargu_fc_t[indext, :]
            sect_labelt = prec_fct[indext]
            nfc_st = torch.cat((sect_fct, nfc_s), 0)
            nnargufc_st = torch.cat((ect_argufct, nargu_fc_s), 0)
            label_st = torch.cat((sect_labelt, label_source), 0)
            labeled_feature_ts = torch.cat([nfc_st.unsqueeze(1), nnargufc_st.unsqueeze(1)], dim=1)
            labeled_loss = criterion(labeled_feature_ts, label_st)
        else:
            labeled_loss = 0.0
        unconloss = criterion(total_feature_target)

        consist_loss = my_loss.mse_loss(cluster_target, fc_t, share_num)

        condition_loss, cluster_loss = my_loss.cond_loss(cluster_source, cluster_target, switch_outputs, argu_cluster_source, argu_cluster_target,
                                                             argu_switch_outputs, share_num, criterion)

        dset_loaders["middle"] = None  
        outputs = torch.cat((fc_s, fc_t), dim=0)
        features = torch.cat((feature_source, feature_target), dim=0)

        instance_weight = my_loss.Intance_weight(cluster_source, network.calc_coeff(i, 1, 0, 10, args.max_iterations))
        instance_weight = instance_weight.detach()

        cls_weight = torch.ones(outputs.size(0)).cuda()
        if class_weight is not None and args.weight_aug:
            cls_weight[0:train_bs] = class_weight[label_source]
            # if dset_loaders["middle"] is not None:
            #     cls_weight[2 * train_bs::] = class_weight[labels_middle]
            # compute source cross-entropy loss
        if class_weight is not None and args.weight_cls:
            src_ = torch.nn.CrossEntropyLoss(reduction='none')(fc_s, label_source)
            weight = class_weight[label_source].detach()
            weight = weight.mul(instance_weight)
            src_loss = torch.sum(weight * src_) / (1e-8 + torch.sum(weight).item())
        else:
            src_loss = torch.nn.CrossEntropyLoss()(fc_s, label_source)

        softmax_out = torch.nn.Softmax(dim=1)(outputs)
        transfer_loss, mean_entropy= loss.HDA_UDA(features, softmax_out, ad_net, network.calc_coeff(i), cls_weight)

        softmax_tar_out = torch.nn.Softmax(dim=1)(fc_t)
        tar_loss = torch.mean(my_loss.Entropy(softmax_tar_out))

        total_loss = src_loss + transfer_loss + 0.1*(unconloss + labeled_loss + cluster_loss + consist_loss)

        if i % config["print_num"] == 0 :

            log_str = "iter:{:05d},transfer:{:.5f},classifier:{:.5f},tarpred_loss:{:.5f}".format(i, transfer_loss, src_loss, tar_loss)
            print(log_str)

        total_loss.backward()
        optimizer.step()



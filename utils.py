#encoding:utf-8
# -----------------------------------------------------------
# "Remote Sensing Cross-Modal Text-Image Retrieval Based on Global and Local Information"
# Yuan, Zhiqiang and Zhang, Wenkai and Changyuan Tian and Xuee, Rong and Zhengyuan Zhang and Wang, Hongqi and Fu, Kun and Sun, Xian
# Writen by YuanZhiqiang, 2021.  Our code is depended on AMFMN
# ------------------------------------------------------------

import torch
import numpy as np
import sys
import  math
from torch.autograd import Variable
from collections import OrderedDict
import torch.nn as nn
import shutil
import time

# 从npy中读取
def load_from_npy(filename):
    info = np.load(filename, allow_pickle=True)
    return info

# 保存结果到txt文件
def log_to_txt( contexts=None,filename="save.txt", mark=False,encoding='UTF-8',mode='a'):
    f = open(filename, mode,encoding=encoding)
    if mark:
        sig = "------------------------------------------------\n"
        f.write(sig)
    elif isinstance(contexts, dict):
        tmp = ""
        for c in contexts.keys():
            tmp += str(c)+" | "+ str(contexts[c]) +"\n"
        contexts = tmp
        f.write(contexts)
    else:
        if isinstance(contexts,list):
            tmp = ""
            for c in contexts:
                tmp += str(c)
            contexts = tmp
        else:
            contexts = contexts + "\n"
        f.write(contexts)


    f.close()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)

def update_values(dict_from, dict_to):
    for key, value in dict_from.items():
        if isinstance(value, dict):
            update_values(dict_from[key], dict_to[key])
        elif value is not None:
            dict_to[key] = dict_from[key]
    return dict_to

def params_count(model):
    count = 0
    for p in model.parameters():
        c = 1
        for i in range(p.dim()):
            c *= p.size(i)
        count += c
    return count


def collect_match(input):
    """change the model output to the match matrix"""
    image_size = input.size(0)
    text_size = input.size(1)

    # match_v = torch.zeros(image_size, text_size, 1)
    # match_v = match_v.view(image_size*text_size, 1)
    input_ = nn.LogSoftmax(2)(input)
    output = torch.index_select(input_, 2, Variable(torch.LongTensor([1])).cuda())

    return output


def collect_neg(input):
    """"collect the hard negative sample"""
    if input.dim() != 2:
        return ValueError

    batch_size = input.size(0)
    mask = Variable(torch.eye(batch_size)>0.5).cuda()
    output = input.masked_fill_(mask, 0)
    output_r = output.max(1)[0]
    output_c = output.max(0)[0]
    loss_n = torch.mean(output_r) + torch.mean(output_c)
    return loss_n

def calcul_loss(scores, size, margin, max_violation=False):
    diagonal = scores.diag().view(size, 1)

    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)

    # compare every diagonal score to scores in its column
    # caption retrieval
    cost_s = (margin + scores - d1).clamp(min=0)
    # compare every diagonal score to scores in its row
    # image retrieval
    cost_im = (margin + scores - d2).clamp(min=0)

    mask = torch.eye(scores.size(0)) > .5
    I = Variable(mask)
    if torch.cuda.is_available():
        I = I.cuda()
    cost_s = cost_s.masked_fill_(I, 0)
    cost_im = cost_im.masked_fill_(I, 0)

    if max_violation:
        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]

    return cost_s.sum() + cost_im.sum()


def acc_train(input):
    predicted = input.squeeze().numpy()
    batch_size = predicted.shape[0]
    predicted[predicted > math.log(0.5)] = 1
    predicted[predicted < math.log(0.5)] = 0
    target = np.eye(batch_size)
    recall = np.sum(predicted * target) / np.sum(target)
    precision = np.sum(predicted * target) / np.sum(predicted)
    acc = 1 - np.sum(abs(predicted - target)) / (target.shape[0] * target.shape[1])

    return acc, recall, precision

def acc_i2t(input):
    """Computes the precision@k for the specified values of k of i2t"""
    #input = collect_match(input).numpy()
    image_size = input.shape[0]
    ranks = np.zeros(image_size)
    # ranks_ = np.zeros(image_size//5)
    top1 = np.zeros(image_size)

    for index in range(image_size):
        inds = np.argsort(input[index])[::-1]
        # Score
        rank = 1e20
        # index_ = index // 5
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]

            if tmp < rank:
                rank = tmp
        if rank == 1e20:
            print('error')
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1)


def acc_t2i(input):
    """Computes the precision@k for the specified values of k of t2i"""
    #input = collect_match(input).numpy()
    image_size = input.shape[0]
    ranks = np.zeros(5*image_size)
    top1 = np.zeros(5*image_size)
    # ranks_ = np.zeros(image_size // 5)
    # --> (5N(caption), N(image))
    input = input.T

    for index in range(image_size):
        for i in range(5):
            inds = np.argsort(input[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]


    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1)

def shard_dis(images, captions, model, shard_size=128, lengths=None):
    """compute image-caption pairwise distance during validation and test"""

    n_im_shard = (len(images) - 1) // shard_size + 1
    n_cap_shard = (len(captions) - 1) // shard_size + 1

    d = np.zeros((len(images), len(captions)))

    for i in range(n_im_shard):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))

#        print("======================")
#        print("im_start:",im_start)
#        print("im_end:",im_end)

        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_distance batch (%d,%d)' % (i,j))
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))

            im = Variable(torch.from_numpy(images[im_start:im_end]), volatile=True).float().cuda()
            s = Variable(torch.from_numpy(captions[cap_start:cap_end]), volatile=True).cuda()
            l = lengths[cap_start:cap_end]

            sim = model(im, s,l)
            sim = sim.squeeze()
            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return d

def acc_i2t2(input):
    """Computes the precision@k for the specified values of k of i2t"""
    #input = collect_match(input).numpy()
    image_size = input.shape[0]
    ranks = np.zeros(image_size)
    top1 = np.zeros(image_size)

    for index in range(image_size):
        inds = np.argsort(input[index])[::-1]
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]


    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1)


def acc_t2i2(input):
    """Computes the precision@k for the specified values of k of t2i"""
    #input = collect_match(input).numpy()
    image_size = input.shape[0]
    ranks = np.zeros(5*image_size)
    top1 = np.zeros(5*image_size)

    # --> (5N(caption), N(image))
    input = input.T

    for index in range(image_size):
        for i in range(5):
            inds = np.argsort(input[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1)

def shard_dis_reg(images, captions, model, shard_size=128, lengths=None):
    """compute image-caption pairwise distance during validation and test"""

    n_im_shard = (len(images) - 1) // shard_size + 1
    n_cap_shard = (len(captions) - 1) // shard_size + 1

    d = np.zeros((len(images), len(captions)))

    for i in range(len(images)):
        # im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
        im_index = i
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_distance batch (%d,%d)' % (i,j))
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))

            s = Variable(torch.from_numpy(captions[cap_start:cap_end]), volatile=True).cuda()
            im = Variable(torch.from_numpy(images[i]), volatile=True).float().unsqueeze(0).expand(len(s), 3, 256, 256).cuda()

            l = lengths[cap_start:cap_end]

            sim = model(im, s, l)[:, 1]



            sim = sim.squeeze()
            d[i, cap_start:cap_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return d

def shard_dis_GaLR(images, input_local_rep, input_local_adj, captions, model, shard_size=128, lengths=None):
    """compute image-caption pairwise distance during validation and test"""

    n_im_shard = (len(images) - 1) // shard_size + 1
    n_cap_shard = (len(captions) - 1) // shard_size + 1

    d = np.zeros((len(images), len(captions))) # score matrix

    all = []
    # @@ for test
    #  d:(452, 2260) n_im_shard:4 n_cap_shard:18
    
    print(f"all shapes utils:\n d:{d.shape} n_im_shard:{n_im_shard} n_cap_shard:{n_cap_shard}")
    print(f"all shapes utils type :\n d:{type(d)}")
    
    for i in range(n_im_shard):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))

        print("======================")
        print("im_start:",im_start)
        print("im_end:",im_end)

        for j in range(n_cap_shard):

            sys.stdout.write('\r>> shard_distance batch (%d,%d)' % (i,j))
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))

            # update for higher version torch
            with torch.no_grad():
                im = Variable(torch.from_numpy(images[im_start:im_end])).float().cuda()
                local_rep = Variable(torch.from_numpy(input_local_rep[im_start:im_end])).float().cuda()
                local_adj = Variable(torch.from_numpy(input_local_adj[im_start:im_end])).float().cuda()
                s = Variable(torch.from_numpy(captions[cap_start:cap_end])).cuda()
                
            # im = Variable(torch.from_numpy(images[im_start:im_end]), volatile=True).float().cuda()
            # local_rep = Variable(torch.from_numpy(input_local_rep[im_start:im_end]), volatile=True).float().cuda()
            # local_adj = Variable(torch.from_numpy(input_local_adj[im_start:im_end]), volatile=True).float().cuda()

            # s = Variable(torch.from_numpy(captions[cap_start:cap_end]), volatile=True).cuda()
            l = lengths[cap_start:cap_end]

            t1 = time.time()
            sim = model(im, local_rep, local_adj, s, l)
            t2 = time.time()
            all.append(t2-t1)
            # sim:<class 'torch.Tensor'> sim:torch.Size([128, 128])
            # sim.data.cpu().numpy():<class 'numpy.ndarray'> sim:(128, 128)
            """a piece of  sim.data :
            [[-0.0066296  -0.00427614 -0.04136446 -0.00036203 -0.02456319]
            [ 0.04639398  0.05223051  0.01300559  0.05274548 -0.01010514]
            [ 0.05061461  0.04705416  0.00953493  0.05804242  0.01446959]
            [ 0.03841243  0.03201586 -0.0086829   0.04708235  0.00733969]
            [ 0.02666652  0.03443651 -0.02278918  0.02741908 -0.00504976]]
            """
            
            print(f"utils sim :\n sim:{type(sim)} sim:{sim.size()}")
            print(f"utils sim.data :\n sim:{type(sim.data.cpu().numpy())} sim:{sim.data.cpu().numpy().shape}")
            print(f"a piece of  sim.data :\n {sim.data.cpu().numpy()[:5,:5]}")

            sim = sim.squeeze()
            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    print("infer time:",np.average(all))
    return d


# ======================================================================================================================================================
# ======================================================================================================================================================
# My model

def shard_dis_GaLR_Sent_Bert(images, input_local_rep, input_local_adj, captions, model,sent_embs_list, shard_size=128, lengths=None):
    """compute image-caption pairwise distance during validation and test"""

    n_im_shard = (len(images) - 1) // shard_size + 1
    n_cap_shard = (len(captions) - 1) // shard_size + 1

    d = np.zeros((len(images), len(captions)))

    all = []

    for i in range(n_im_shard):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))

        print("======================")
        print("im_start:",im_start)
        print("im_end:",im_end)

        for j in range(n_cap_shard):

            sys.stdout.write('\r>> shard_distance batch (%d,%d)' % (i,j))
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))

            # update for higher version torch
            with torch.no_grad():
                im = Variable(torch.from_numpy(images[im_start:im_end])).float().cuda()
                local_rep = Variable(torch.from_numpy(input_local_rep[im_start:im_end])).float().cuda()
                local_adj = Variable(torch.from_numpy(input_local_adj[im_start:im_end])).float().cuda()
                s = Variable(torch.from_numpy(captions[cap_start:cap_end])).cuda()
                semb = Variable(torch.from_numpy(sent_embs_list[cap_start:cap_end])).cuda()

            l = lengths[cap_start:cap_end]

            t1 = time.time()
            # input_visual, input_local_rep, input_local_adj, input_text,  sent_embs, lengths
            sim = model(im, local_rep, local_adj, s,semb, l)
            t2 = time.time()
            all.append(t2-t1)

            sim = sim.squeeze()
            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    print("infer time:",np.average(all))
    return d

# @@ bert embedding for caption
def shard_dis_GaLR_Bert(images, input_local_rep, input_local_adj, captions, model, input_ids, token_type_ids, attention_mask, shard_size=128, lengths=None):
    """compute image-caption pairwise distance during validation and test in bert embedding"""

    n_im_shard = (len(images) - 1) // shard_size + 1
    n_cap_shard = (len(captions) - 1) // shard_size + 1

    d = np.zeros((len(images), len(captions)))

    all = []
    #  d:(452, 2260) input_ids:torch.Size([157200, 47]) n_im_shard:4 n_cap_shard:18 
    # all shapes utils type : d:<class 'numpy.ndarray'> input_ids:<class 'torch.Tensor'> token_type_ids:<class 'torch.Tensor'> attention_mask:<class 'torch.Tensor'>
    # print(f"all shapes utils:\n d:{d.shape} input_ids:{input_ids.shape} n_im_shard:{n_im_shard} n_cap_shard:{n_cap_shard}")
    # print(f"all shapes utils type :\n d:{type(d)} input_ids:{type(input_ids)} token_type_ids:{type(token_type_ids)} attention_mask:{type(attention_mask)}")
    
    for i in range(n_im_shard):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))

        print("======================")
        print("im_start:",im_start)
        print("im_end:",im_end)

        for j in range(n_cap_shard):

            sys.stdout.write('\r>> shard_distance batch (%d,%d)' % (i,j))
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))

            # update for higher version torch
            with torch.no_grad():
                # print("@@ utils \n{}\n{}\n{}".format(type(images),type(captions),type(input_ids)))
                # print("@@ utils \n{}\n{}\n{}".format(np.size(images),np.size(captions),input_ids.size()))

                im = Variable(torch.from_numpy(images[im_start:im_end])).float().cuda()
                local_rep = Variable(torch.from_numpy(input_local_rep[im_start:im_end])).float().cuda()
                local_adj = Variable(torch.from_numpy(input_local_adj[im_start:im_end])).float().cuda()
                s = Variable(torch.from_numpy(captions[cap_start:cap_end])).cuda()
                
                iidx = Variable(torch.from_numpy(input_ids[cap_start:cap_end])).cuda()
                ttids = Variable(torch.from_numpy(token_type_ids[cap_start:cap_end])).cuda()
                amask = Variable(torch.from_numpy(attention_mask[cap_start:cap_end])).cuda()
                
            # iidx:<class 'torch.Tensor'> iidx:torch.Size([128, 47])
            # iidx:<class 'torch.Tensor'> iidx:torch.Size([128, 768, 47])  im:<class 'torch.Tensor'> iidx:torch.Size([128, 3, 256, 256])  captions:<class 'torch.Tensor'> iidx:torch.Size([128, 47])
            
            # print(f"all utils:\n iidx:{type(iidx)} iidx:{iidx.shape}")
            # print(f"all utils:\n im:{type(im)} iidx:{im.shape}")
            # print(f"all utils:\n captions:{type(s)} iidx:{s.shape}")
                
            # print("@@@ utils \n{}\n{}\n{}".format(type(im),type(s),type(iidx)))
            # print("@@@ utils \n{}\n{}\n{}".format(im.size(),s.size(),iidx.size()))
                
                
            # im = Variable(torch.from_numpy(images[im_start:im_end]), volatile=True).float().cuda()
            # local_rep = Variable(torch.from_numpy(input_local_rep[im_start:im_end]), volatile=True).float().cuda()
            # local_adj = Variable(torch.from_numpy(input_local_adj[im_start:im_end]), volatile=True).float().cuda()

            # s = Variable(torch.from_numpy(captions[cap_start:cap_end]), volatile=True).cuda()
            l = lengths[cap_start:cap_end]

            t1 = time.time()
            sim = model(im, local_rep, local_adj, s, iidx,ttids,amask, l)
            t2 = time.time()
            all.append(t2-t1)
            # sim:<class 'torch.Tensor'> sim:torch.Size([128, 128])
            # sim.data.cpu().numpy():<class 'numpy.ndarray'> sim:(128, 128)
            """a piece of  sim.data :
             [[ 0.68847895 -0.23251936 -0.75733054 -0.5951389  -0.72028965]
            [ 0.6842005   0.07729551 -0.86723685 -0.37875015 -0.3028088 ]
            [ 0.39638546 -0.08011446 -0.5635441  -0.4696055  -0.62302715]
            [-0.24134123  0.01998806 -0.11424804 -0.20329574  0.3711012 ]
            [-0.16456978 -0.03087491 -0.9853229  -0.5842093   0.21048312]]

            """
            
            # print(f"utils sim :\n sim:{type(sim)} sim:{sim.size()}")
            # print(f"utils sim.data :\n sim:{type(sim.data.cpu().numpy())} sim:{sim.data.cpu().numpy().shape}")
            # print(f"a piece of  sim.data :\n {sim.data.cpu().numpy()[:5,:5]}")
            sim = sim.squeeze()
            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()

    sys.stdout.write('\n')
    print("infer time:",np.average(all))
    return d

def save_checkpoint(state, is_best, filename, prefix='', model_name = None):
    tries = 15
    error = None

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            # torch.save(state, prefix + filename)
            if is_best:
                torch.save(state, prefix +model_name +'_best.pth.tar')

        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error

def adjust_learning_rate(options, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        lr = param_group['lr']

        if epoch % options['optim']['lr_update_epoch'] == options['optim']['lr_update_epoch'] - 1:
            lr = lr * options['optim']['lr_decay_param']

        param_group['lr'] = lr

    print("Current lr: {}".format(optimizer.state_dict()['param_groups'][0]['lr']))

def load_from_txt(filename, encoding="utf-8"):
    f = open(filename,'r' ,encoding=encoding)
    contexts = f.readlines()
    return contexts

#encoding:utf-8
# -----------------------------------------------------------
# 将bert替换原始的skipthoughts+gru文本编码方式
# ------------------------------------------------------------

import time
import torch
import numpy as np
import sys
from torch.autograd import Variable
import utils
import tensorboard_logger as tb_logger

# from torch.utils.tensorboard import SummaryWriter

import logging
from torch.nn.utils.clip_grad import clip_grad_norm


def show_model(train_loader, model, optimizer, epoch, opt={}):

    # extract value
    grad_clip = opt['optim']['grad_clip']
    max_violation = opt['optim']['max_violation']
    margin = opt['optim']['margin']
    loss_name = opt['model']['name'] + "_" + opt['dataset']['datatype']
    print_freq = opt['logs']['print_freq']

    # switch to train mode
    model.train()
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    train_logger = utils.LogCollector()

    end = time.time()
    params = list(model.parameters())
    for i, train_data in enumerate(train_loader):
        images, local_rep, local_adj, captions, lengths, ids= train_data

        batch_size = images.size(0)
        margin = float(margin)
        # measure data loading time
        data_time.update(time.time() - end)
        model.logger = train_logger

        input_visual = Variable(images)
        input_local_rep = Variable(local_rep)
        input_local_adj = Variable(local_adj)

        input_text = Variable(captions)

        if torch.cuda.is_available():
            input_visual = input_visual.cuda()
            input_local_rep = input_local_rep.cuda()
            input_local_adj = input_local_adj.cuda()

            input_text = input_text.cuda()
        
        # train model here  
        # scores = model(input_visual, input_local_rep, input_local_adj, input_text, lengths)
            
        # 遍历模型中的所有层
        for name, module in model.named_children():
            print(f"Layer: {name}")
            print(f"module: {module}")
            # output_tensor = module(input_visual, input_local_rep, input_local_adj, input_text, lengths)
            # print(f"Output shape: {output_tensor.shape}")
            # input_visual, input_local_rep, input_local_adj, input_text, lengths = output_tensor


def train(train_loader, model, optimizer, epoch, opt={}):

    # extract value
    grad_clip = opt['optim']['grad_clip']
    max_violation = opt['optim']['max_violation']
    margin = opt['optim']['margin']
    loss_name = opt['model']['name'] + "_" + opt['dataset']['datatype']
    print_freq = opt['logs']['print_freq']

    # switch to train mode
    model.train()
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    train_logger = utils.LogCollector()

    end = time.time()
    params = list(model.parameters())
    for i, train_data in enumerate(train_loader):
        images, local_rep, local_adj, captions, lengths, ids,\
            input_ids, token_type_ids, attention_mask= train_data

        batch_size = images.size(0)
        margin = float(margin)
        # measure data loading time
        data_time.update(time.time() - end)
        model.logger = train_logger

        input_visual = Variable(images)
        input_local_rep = Variable(local_rep)
        input_local_adj = Variable(local_adj)

        input_text = Variable(captions)

        if torch.cuda.is_available():
            input_visual = input_visual.cuda()
            input_local_rep = input_local_rep.cuda()
            input_local_adj = input_local_adj.cuda()

            input_text = input_text.cuda()
            
            
            # @@ bert encoding
            input_ids = input_ids.cuda()
            token_type_ids = token_type_ids.cuda()
            attention_mask = attention_mask.cuda()
        
        # print("engine  input_ids", type(input_ids))
        # print("token_type_ids", type(token_type_ids))
        # print("attention_mask", type(attention_mask))

        
        # raise NotImplementedError
        # @@ train model here  
        scores = model(input_visual, input_local_rep, input_local_adj, input_text, 
                        input_ids, token_type_ids, attention_mask, lengths)
        # img, input_local_rep, input_local_adj, text, input_ids, 
        #         token_type_ids, attention_mask,text_lens    
            
        torch.cuda.synchronize()
        loss = utils.calcul_loss(scores, input_visual.size(0), margin, max_violation=max_violation, )

        if grad_clip > 0:
            clip_grad_norm(params, grad_clip)

        train_logger.update('L', loss.cpu().data.numpy())


        optimizer.zero_grad()
        loss.backward()
        torch.cuda.synchronize()
        optimizer.step()
        torch.cuda.synchronize()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f}\t'
                '{elog}\t'
                .format(epoch, i, len(train_loader),
                        batch_time=batch_time,
                        elog=str(train_logger)))

            utils.log_to_txt(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f}\t'
                '{elog}\t'
                    .format(epoch, i, len(train_loader),
                            batch_time=batch_time,
                            elog=str(train_logger)),
                opt['logs']['ckpt_save_path']+ opt['model']['name'] + "_" + opt['dataset']['datatype'] +".txt"
            )
        # SummaryWriter.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        train_logger.tb_log(tb_logger, step=model.Eiters)


def validate(val_loader, model):

    model.eval()
    val_logger = utils.LogCollector()
    model.logger = val_logger

    start = time.time()
    input_visual = np.zeros((len(val_loader.dataset), 3, 256, 256))
    input_local_rep = np.zeros((len(val_loader.dataset), 20, 20))
    input_local_adj = np.zeros((len(val_loader.dataset), 20, 20))

    input_text = np.zeros((len(val_loader.dataset), 47), dtype=np.int64) # fix len 47 for nltk
    input_text_lengeth = [0]*len(val_loader.dataset)
    
    # @@ bert list
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    for i, val_data in enumerate(val_loader):

        images, local_rep, local_adj, captions, lengths, ids, \
            input_ids, token_type_ids, attention_mask= val_data
        
        input_ids_list.append(input_ids)
        token_type_ids_list.append(token_type_ids)
        attention_mask_list.append(attention_mask)
        
        # 
        for (id, img,rep,adj, cap, l , iids, ttids, amask) in \
            zip(ids, (images.numpy().copy()),(local_rep.numpy().copy()),\
                (local_adj.numpy().copy()), (captions.numpy().copy()), lengths,\
                    input_ids.numpy().copy(), token_type_ids.numpy().copy(), attention_mask.numpy().copy()):
            input_visual[id] = img
            input_local_rep[id] = rep
            input_local_adj[id] = adj

            input_text[id, :captions.size(1)] = cap
            input_text_lengeth[id] = l
            
    
    # print("@@ mye_engine \n{}\n{}\n{}".format(np.size(input_text),np.size(input_visual),input_ids.size()))
   
    # raise NotImplementedError
    input_ids_list=torch.cat(input_ids_list, dim=0)
    token_type_ids_list=torch.cat(token_type_ids_list, dim=0)
    attention_mask_list=torch.cat(attention_mask_list, dim=0)
    
    input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)])
    input_local_rep = np.array([input_local_rep[i] for i in range(0, len(input_local_rep), 5)])
    input_local_adj = np.array([input_local_adj[i] for i in range(0, len(input_local_adj), 5)])
    # print("@@ mye_engine2 \n{}\n{}\n{}".format(np.size(input_text),np.size(input_visual),input_ids_list.size()))

    d = utils.shard_dis_GaLR_Bert(input_visual, input_local_rep, input_local_adj, input_text, model, \
        input_ids_list, token_type_ids_list, attention_mask_list, lengths=input_text_lengeth)

    end = time.time()
    print("calculate similarity time:", end - start)

    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t2(d)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanri))
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_t2i2(d)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1t, r5t, r10t, medrt, meanrt))
    currscore = (r1t + r5t + r10t + r1i + r5i + r10i)/6.0

    all_score = "r1i:{} r5i:{} r10i:{} medri:{} meanri:{}\n r1t:{} r5t:{} r10t:{} medrt:{} meanrt:{}\n sum:{}\n ------\n".format(
        r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, currscore
    )
  
    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanri', meanri, step=model.Eiters)
    tb_logger.log_value('r1t', r1t, step=model.Eiters)
    tb_logger.log_value('r5t', r5t, step=model.Eiters)
    tb_logger.log_value('r10t', r10t, step=model.Eiters)
    tb_logger.log_value('medrt', medrt, step=model.Eiters)
    tb_logger.log_value('meanrt', meanrt, step=model.Eiters)
    tb_logger.log_value('rsum', currscore, step=model.Eiters)

    return currscore, all_score


def validate_test(val_loader, model):
    model.eval()
    val_logger = utils.LogCollector()
    model.logger = val_logger

    start = time.time()
    input_visual = np.zeros((len(val_loader.dataset), 3, 256, 256))
    input_local_rep = np.zeros((len(val_loader.dataset), 20, 20))
    input_local_adj = np.zeros((len(val_loader.dataset), 20, 20))

    input_text = np.zeros((len(val_loader.dataset), 47), dtype=np.int64)
    input_text_lengeth = [0] * len(val_loader.dataset)

    embed_start = time.time()
    
    # @@ bert list
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    
    for i, val_data in enumerate(val_loader):

        images,local_rep, local_adj, captions, lengths, ids, \
            input_ids, token_type_ids, attention_mask= val_data

        for (id, img,rep,adj, cap, l) in zip(ids, \
            (images.numpy().copy()),(local_rep.numpy().copy()),(local_adj.numpy().copy()), (captions.numpy().copy()), lengths):
            input_visual[id] = img
            input_local_rep[id] = rep
            input_local_adj[id] = adj
                
            input_ids_list.append(input_ids)
            token_type_ids_list.append(token_type_ids)
            attention_mask_list.append(attention_mask)

            input_text[id, :captions.size(1)] = cap
            input_text_lengeth[id] = l

    # raise NotImplementedError
    input_ids_list=torch.cat(input_ids_list, dim=0)
    token_type_ids_list=torch.cat(token_type_ids_list, dim=0)
    attention_mask_list=torch.cat(attention_mask_list, dim=0)
    
    input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)])
    input_local_rep = np.array([input_local_rep[i] for i in range(0, len(input_local_rep), 5)])
    input_local_adj = np.array([input_local_adj[i] for i in range(0, len(input_local_adj), 5)])
    embed_end = time.time()
    print("embedding time: {}".format(embed_end-embed_start))

    d = utils.shard_dis_GaLR_Bert(input_visual, input_local_rep, input_local_adj, input_text, model,\
        input_ids_list, token_type_ids_list, attention_mask_list,lengths=input_text_lengeth)

    end = time.time()
    print("calculate similarity time:", end - start)

    return d

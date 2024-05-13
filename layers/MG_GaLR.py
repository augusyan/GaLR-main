import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
from .all_utils import *
import copy
import ast

    """
    TODO:
    1. Bert的sentence-level和token-level
    2. MIDF的动态部分，在textual部分增加动态


    """
    
class Fusion_MIDF(nn.Module):
    def __init__(self, opt):
        super(Fusion_MIDF, self).__init__()
        self.opt = opt

        # local trans
        self.l2l_SA = SA(opt)

        # global trans
        self.g2g_SA = SA(opt)

        # local correction
        self.g2l_SGA = SGA(opt)

        # global supplement
        self.l2g_SGA = SGA(opt)

        # dynamic fusion
        self.dynamic_weight = nn.Sequential(
            nn.Linear(opt['embed']['embed_dim'], opt['fusion']['dynamic_fusion_dim']),
            nn.Sigmoid(),
            nn.Dropout(p=opt['fusion']['dynamic_fusion_drop']),
            nn.Linear(opt['fusion']['dynamic_fusion_dim'], 2),
            nn.Softmax()
        )

    def forward(self, global_feature, local_feature):

        global_feature = torch.unsqueeze(global_feature, dim=1)
        local_feature = torch.unsqueeze(local_feature, dim=1)

        # global trans
        global_feature = self.g2g_SA(global_feature)
        # local trans
        local_feature = self.l2l_SA(local_feature)

        # local correction
        local_feature = self.g2l_SGA(local_feature, global_feature)

        # global supplement
        global_feature = self.l2g_SGA(global_feature, local_feature)

        global_feature_t = torch.squeeze(global_feature, dim=1)
        local_feature_t = torch.squeeze(local_feature, dim=1)

        global_feature = F.sigmoid(local_feature_t) * global_feature_t
        local_feature = global_feature_t + local_feature_t

        # dynamic fusion
        feature_gl = global_feature + local_feature
        dynamic_weight = self.dynamic_weight(feature_gl)

        weight_global = dynamic_weight[:, 0].reshape(feature_gl.shape[0],-1).expand_as(global_feature)


        weight_local = dynamic_weight[:, 0].reshape(feature_gl.shape[0],-1).expand_as(global_feature)

        visual_feature = weight_global*global_feature + weight_local*local_feature

        return visual_feature

    
class BaseModel(nn.Module):
    def __init__(self, opt={}, vocab_words=[]):
        super(BaseModel, self).__init__()

        # img feature
        self.extract_feature = ExtractFeature(opt = opt)
        self.drop_g_v = nn.Dropout(0.3)

        # vsa feature
        self.mvsa =VSA_Module(opt = opt)

        # local feature
        self.local_feature = GCN()
        self.drop_l_v = nn.Dropout(0.3)

        # text feature
        ## TODO:1. using sentence_bert and bert for global and local text feature
        self.text_feature = Skipthoughts_Embedding_Module(
            vocab= vocab_words,
            opt = opt
        )

        # fusion
        self.fusion = Fusion_MIDF(opt = opt)

        # weight
        self.gw = opt['global_local_weight']['global']
        self.lw = opt['global_local_weight']['local']

        self.Eiters = 0

    def forward(self, img, input_local_rep, input_local_adj, text, text_lens=None):

        # extract features, from different layers of ResNet
        lower_feature, higher_feature, solo_feature = self.extract_feature(img)

        # mvsa featrues, to gain masked remote sensing feature from mvsa mechnism
        global_feature = self.mvsa(lower_feature, higher_feature, solo_feature)
#        global_feature = solo_feature

        # extract local feature
        local_feature = self.local_feature(input_local_adj, input_local_rep)
        
        # dynamic fusion @@@
        visual_feature = self.fusion(global_feature, local_feature)

        # text features
        text_feature = self.text_feature(text)

        sims = cosine_sim(visual_feature, text_feature)
        #sims = cosine_sim(self.lw*self.drop_l_v(local_feature) + self.gw*self.drop_g_v(global_feature), text_feature)
        return sims

def factory(opt, vocab_words, cuda=True, data_parallel=True):
    opt = copy.copy(opt)

    model = BaseModel(opt, vocab_words)

    if data_parallel:
        model = nn.DataParallel(model).cuda()
        if not cuda:
            raise ValueError

    if cuda:
        model.cuda()

    return model


class Defusion_MIDF(nn.Module):
    def __init__(self, opt):
        super(Defusion_MIDF, self).__init__()
        self.opt = opt

        # local trans
        self.l2l_SA = SA(opt)

        # global trans
        self.g2g_SA = SA(opt)

        # local correction
        self.g2l_SGA = SGA(opt)

        # global supplement
        self.l2g_SGA = SGA(opt)

        # dynamic fusion
        self.dynamic_weight = nn.Sequential(
            nn.Linear(opt['embed']['embed_dim'], opt['fusion']['dynamic_fusion_dim']),
            nn.Sigmoid(),
            nn.Dropout(p=opt['fusion']['dynamic_fusion_drop']),
            nn.Linear(opt['fusion']['dynamic_fusion_dim'], 2),
            nn.Softmax()
        )

    def forward(self, global_feature, local_feature):

        global_feature = torch.unsqueeze(global_feature, dim=1)
        local_feature = torch.unsqueeze(local_feature, dim=1)

        # global trans
        global_feature = self.g2g_SA(global_feature)
        # local trans
        local_feature = self.l2l_SA(local_feature)

        # local correction
        local_feature = self.g2l_SGA(local_feature, global_feature)

        # global supplement
        global_feature = self.l2g_SGA(global_feature, local_feature)

        global_feature_t = torch.squeeze(global_feature, dim=1)
        local_feature_t = torch.squeeze(local_feature, dim=1)

        global_feature = F.sigmoid(local_feature_t) * global_feature_t
        local_feature = global_feature_t + local_feature_t

        # dynamic fusion
        feature_gl = global_feature + local_feature
        dynamic_weight = self.dynamic_weight(feature_gl)

        weight_global = dynamic_weight[:, 0].reshape(feature_gl.shape[0],-1).expand_as(global_feature)


        weight_local = dynamic_weight[:, 0].reshape(feature_gl.shape[0],-1).expand_as(global_feature)

        #  weight_global.size(): torch.Size([100, 512])   weight_local.size(): torch.Size([100, 512]) visual_feature: torch.Size([100, 512])
        # print(f"@@@@ weight_global {weight_global.size()}  weight_local {weight_local.size()}")
        visual_feature = weight_global*global_feature + weight_local*local_feature
        # print(f"viual_feature {weight_global.size()}")

        return weight_global, weight_local

class DualModel(nn.Module):
    def __init__(self, opt={}, vocab_words=[]):
        super(DualModel, self).__init__()

        # img feature
        self.extract_feature = ExtractFeature(opt = opt)
        self.drop_g_v = nn.Dropout(0.3)

        # vsa feature
        self.mvsa =VSA_Module(opt = opt)

        # local feature
        self.local_feature = GCN()
        self.drop_l_v = nn.Dropout(0.3)

        # text feature
        ## TODO:1. using sentence_bert and bert for global and local text feature
        self.text_feature = Skipthoughts_Embedding_Module(
            vocab= vocab_words,
            opt = opt
        )

        # fusion
        self.fusion = Fusion_MIDF(opt = opt)
        self.defusion = Defusion_MIDF(opt = opt)

        # weight
        self.gw = opt['global_local_weight']['global']
        self.lw = opt['global_local_weight']['local']

        self.Eiters = 0

    def forward(self, img, input_local_rep, input_local_adj, text, text_lens=None):

        # extract features, from different layers of ResNet
        lower_feature, higher_feature, solo_feature = self.extract_feature(img)

        # mvsa featrues, to gain masked remote sensing feature from mvsa mechnism
        global_feature = self.mvsa(lower_feature, higher_feature, solo_feature)
#        global_feature = solo_feature

        # extract local feature
        local_feature = self.local_feature(input_local_adj, input_local_rep)
        
        # dynamic fusion @@@
        visual_feature = self.fusion(global_feature, local_feature)
        
        golbal_v_feat, local_v_feat = self.defusion(global_feature, local_feature)

        # text features
        text_feature = self.text_feature(text)

        sims_merged = cosine_sim(visual_feature, text_feature)
        
        sims_local = cosine_sim(local_v_feat, text_feature)
        sims_global = cosine_sim(golbal_v_feat, text_feature)
        #sims = cosine_sim(self.lw*self.drop_l_v(local_feature) + self.gw*self.drop_g_v(global_feature), text_feature)
        sims = sims_merged + sims_local + sims_global
        
        return sims

def myfactory(opt, vocab_words, cuda=True, data_parallel=True):
    opt = copy.copy(opt)

    model = DualModel(opt, vocab_words)

    if data_parallel:
        model = nn.DataParallel(model).cuda()
        if not cuda:
            raise ValueError

    if cuda:
        model.cuda()

    return model

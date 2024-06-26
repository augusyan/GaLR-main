import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
from .all_utils import SA, SGA,VSA_Module, ExtractFeature, GCN, dot_product, Skipthoughts_Embedding_Module, cosine_sim,Text_Sent_Embedding_Module,\
Text_token_Embedding_Module,SentBert_Embedding_Module,TextLevelsEmbeddingModule
import torch.nn.functional as F

import copy
import ast

"""
Finished:
1. Bert的sentence-level替换原来的文本编码


TODO:
1.Bert token-level
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

        # extract features
        lower_feature, higher_feature, solo_feature = self.extract_feature(img)

        # mvsa featrues
        global_feature = self.mvsa(lower_feature, higher_feature, solo_feature)
#        global_feature = solo_feature

        # extract local feature
        local_feature = self.local_feature(input_local_adj, input_local_rep)
        
        # dynamic fusion
        visual_feature = self.fusion(global_feature, local_feature)

        # text features
        text_feature = self.text_feature(text)

        sims = cosine_sim(visual_feature, text_feature)
        #sims = cosine_sim(self.lw*self.drop_l_v(local_feature) + self.gw*self.drop_g_v(global_feature), text_feature)
        return sims
    
class MG_GaLR(nn.Module):
    def __init__(self, opt={}, vocab_words=[]):
        super(MG_GaLR, self).__init__()

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
        # self.text_feature = Skipthoughts_Embedding_Module(
        #         vocab= vocab_words,
        #         opt = opt
        #     )
        self.text_encoder = Text_token_Embedding_Module(opt = opt)
        
        # fusion
        self.fusion = Fusion_MIDF(opt = opt)
        # self.defusion = Defusion_MIDF(opt = opt)

        # weight
        # self.gw = opt['global_local_weight']['global']
        # self.lw = opt['global_local_weight']['local']

        self.Eiters = 0

    def forward(self, img, input_local_rep, input_local_adj, text, input_ids, 
                token_type_ids, attention_mask,text_lens=None):

        # extract features, from different layers of ResNet
        lower_feature, higher_feature, solo_feature = self.extract_feature(img)

        # mvsa featrues, to gain masked remote sensing feature from mvsa mechnism
        global_feature = self.mvsa(lower_feature, higher_feature, solo_feature)
        # global_feature = solo_feature

        # extract local feature
        local_feature = self.local_feature(input_local_adj, input_local_rep)
        
        # dynamic fusion @@@
        visual_feature = self.fusion(global_feature, local_feature)
        
        # golbal_v_feat, local_v_feat = self.defusion(global_feature, local_feature)

        # @@@ text features    
        text_feature= self.text_encoder(input_ids, token_type_ids=token_type_ids,
                                        attention_mask=attention_mask)    
        # text_feature = self.text_feature(text)
        # local_t_feat = self.text_feature(text)
        # golbal_t_feat = self.text_feature(text)

        # @@@ similarity
        # sims_merged = dot_product(visual_feature, text_feature)
        sims_merged = cosine_sim(visual_feature, text_feature)
        
        # sims_local = cosine_sim(local_v_feat, local_t_feat)
        # sims_global = cosine_sim(golbal_v_feat, golbal_t_feat)
        sims = sims_merged
        # sims = sims_merged + sims_local + sims_global
        
        return sims

class MG_GaLR_sentbert(nn.Module):
    def __init__(self, opt={}, vocab_words=[]):
        super(MG_GaLR_sentbert, self).__init__()

        # img feature
        self.extract_feature = ExtractFeature(opt = opt)
        self.drop_g_v = nn.Dropout(0.3)

        # vsa feature
        self.mvsa =VSA_Module(opt = opt)

        # local feature
        self.local_feature = GCN()
        self.drop_l_v = nn.Dropout(0.3)

        # text feature
        self.text_encoder = SentBert_Embedding_Module(opt = opt)
        
        # fusion
        self.fusion = Fusion_MIDF(opt = opt)
        # self.defusion = Defusion_MIDF(opt = opt)

        self.Eiters = 0

    # input_visual, input_local_rep, input_local_adj, input_text,  sent_embs, lengths
    def forward(self, img, input_local_rep, input_local_adj, text, sent_embs ,text_lens=None):

        # extract features, from different layers of ResNet
        lower_feature, higher_feature, solo_feature = self.extract_feature(img)

        # mvsa featrues, to gain masked remote sensing feature from mvsa mechnism
        global_feature = self.mvsa(lower_feature, higher_feature, solo_feature)
        # global_feature = solo_feature

        # extract local feature
        local_feature = self.local_feature(input_local_adj, input_local_rep)
        
        # dynamic fusion @@@
        visual_feature = self.fusion(global_feature, local_feature)
        
        # golbal_v_feat, local_v_feat = self.defusion(global_feature, local_feature)

        # @@@ text features    
        text_feature= self.text_encoder(sent_embs)    
        # text_feature = self.text_feature(text)
        # local_t_feat = self.text_feature(text)
        # golbal_t_feat = self.text_feature(text)

        # @@@ similarity
        # sims_merged = dot_product(visual_feature, text_feature)
        sims_merged = cosine_sim(visual_feature, text_feature)
        
        # sims_local = cosine_sim(local_v_feat, local_t_feat)
        # sims_global = cosine_sim(golbal_v_feat, golbal_t_feat)
        sims = sims_merged
        # sims = sims_merged + sims_local + sims_global
        
        return sims

class MG_GaLR_SA(nn.Module):
    def __init__(self, opt={}, vocab_words=[]):
        super(MG_GaLR_SA, self).__init__()

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
        # self.text_feature = Skipthoughts_Embedding_Module(
        #         vocab= vocab_words,
        #         opt = opt
        #     )
        self.text_encoder = TextLevelsEmbeddingModule(opt = opt)
        
        # fusion
        self.fusion = Fusion_MIDF(opt = opt)
        self.defusion = Defusion_MIDF(opt = opt)

        # weight
        self.gw = opt['global_local_weight']['global']
        self.lw = opt['global_local_weight']['local']

        self.Eiters = 0

    def forward(self, img, input_local_rep, input_local_adj, text, input_ids, 
                token_type_ids, attention_mask,text_lens=None):

        # extract features, from different layers of ResNet
        lower_feature, higher_feature, solo_feature = self.extract_feature(img)

        # mvsa featrues, to gain masked remote sensing feature from mvsa mechnism
        global_feature = self.mvsa(lower_feature, higher_feature, solo_feature)
        # global_feature = solo_feature

        # extract local feature
        local_feature = self.local_feature(input_local_adj, input_local_rep)
        
        # dynamic fusion @@@
        visual_feature = self.fusion(global_feature, local_feature)
        
        golbal_v_feat, local_v_feat = self.defusion(global_feature, local_feature)

        # @@@ text features    
        text_feature, sa_text_feature = self.text_encoder(input_ids, token_type_ids=token_type_ids,
                                        attention_mask=attention_mask)    
        # text_feature = self.text_feature(text)
        # local_t_feat = self.text_feature(text)
        # golbal_t_feat = self.text_feature(text)

        # @@@ similarity
        # sims_merged = dot_product(visual_feature, text_feature)
        # sims_merged = cosine_sim(visual_feature, text_feature)
        
        sims_merged = cosine_sim(torch.cat((visual_feature, local_v_feat), dim=1), torch.cat((text_feature, sa_text_feature), dim=1))
        sims = sims_merged
        # sims_local = cosine_sim(local_v_feat, sa_text_feature)
        # sims_global = cosine_sim(golbal_v_feat, golbal_t_feat)
        
        
        # sims = sims_merged + 0.1*sims_local
        # sims = sims_merged + 0.1*sims_local
        # sims = sims_merged + sims_local + sims_global
        print(f"\n input_ids: {input_ids[0,:33]}")
        print(f"\n token_type_ids: {token_type_ids[0,:33]}")
        print(f"\n attention_mask: {attention_mask[0,:33]}")
        print(f"\n sims_merged: {sims_merged[:5,:5]}")
        return sims

def myfactory(opt, vocab_words, cuda=True, data_parallel=True):
    opt = copy.copy(opt)

    # @@ add alternative text encoder
    # model = MG_GaLR(opt, vocab_words)
    if opt['name']=='MG_GaLR_sentbert':
        model = MG_GaLR_sentbert(opt, vocab_words)
    elif opt['name']=='MG_GaLR':
        model = MG_GaLR(opt, vocab_words)
    elif opt['name']=='MG_GaLR_SA':
        model = MG_GaLR_SA(opt, vocab_words)
    else:
        raise ValueError('Invalid model name')

    if data_parallel:
        model = nn.DataParallel(model).cuda()
        if not cuda:
            raise ValueError

    if cuda:
        model.cuda()

    return model



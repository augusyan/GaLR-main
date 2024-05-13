# -----------------------------------------------------------
# "Remote Sensing Cross-Modal Text-Image Retrieval Based on Global and Local Information"
# Yuan, Zhiqiang and Zhang, Wenkai and Changyuan Tian and Xuee, Rong and Zhengyuan Zhang and Wang, Hongqi and Fu, Kun and Sun, Xian
# Writen by YuanZhiqiang, 2021.  Our code is depended on AMFMN
# ------------------------------------------------------------

# A revision version from Skip-thoughs
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import skipthoughts
from skipthoughts import BayesianUniSkip

# Define the factory function to create the seq2vec model, extendable to other models
def factory(vocab_words, opt , dropout=0.25):
    """
    根据给定的参数创建一个序列到向量编码模型。

    Args:
    vocab_words: 词汇表中的单词列表
    opt: 包含架构类型和其他模型参数的字典
    dropout: 可选参数，表示dropout概率，默认值为0.25

    Returns:
    创建的序列到向量编码模型
    """
    if opt['arch'] == 'skipthoughts':
        st_class = getattr(skipthoughts, opt['type'])
        seq2vec = st_class(opt['dir_st'],
                           vocab_words,
                           dropout=dropout,
                           fixed_emb=opt['fixed_emb'])

    else:
        raise NotImplementedError
    return seq2vec

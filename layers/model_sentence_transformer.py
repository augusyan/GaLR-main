import sys, json, random
import torch
import os
import numpy as np
import logging
# import matplotlib
import pandas as pd
from sentence_transformers import SentenceTransformer



def generate_attribute(train_label, test_label, property_path='./data/wiki_train_new_property.npy',
                       prop_list_path='./resources/property_list.html'):
    # # 因为train和test没有交际，已弃用，因为每次读取id可能变化
    # if os.path.exists(property_path):
    #     property2idx, idx2property, pid2vec = np.load(property_path,allow_pickle=True)
    #     return property2idx, idx2property, pid2vec

    property2idx = {}
    idx2property = {}  # {0:'P991',1:'P161'...}
    idx = 0
    for i in set(train_label):
        property2idx[i] = idx
        idx2property[idx] = i
        idx += 1
    for i in set(test_label):
        property2idx[i] = idx
        idx2property[idx] = i
        idx += 1

    prop_list = pd.read_html(prop_list_path)[0]
    prop_list = prop_list.loc[prop_list.ID.isin(property2idx.keys())]
    encoder = SentenceTransformer('../DataAll/all-mpnet-base-v2') #专门提取类别描述的文本
    # encoder = SentenceTransformer('../DataAll/bert-large-nli-mean-tokens')
    # logger.info(f"describution of each relation {prop_list.description.to_list()}")
    # 生成每个类别的描述，wiki_data是113种
    sentence_embeddings = encoder.encode(prop_list.description.to_list())

    pid2vec = {}  # {'P17':array[]...}
    for pid, embedding in zip(prop_list.ID, sentence_embeddings):
        pid2vec[pid] = embedding.astype('float32')
    np.save('./data/wiki_train_new_property.npy', [property2idx, idx2property, pid2vec])

    return property2idx, idx2property, pid2vec
# -----------------------------------------------------------
# "Remote Sensing Cross-Modal Text-Image Retrieval Based on Global and Local Information"
# Yuan, Zhiqiang and Zhang, Wenkai and Changyuan Tian and Xuee, Rong and Zhengyuan Zhang and Wang, Hongqi and Fu, Kun and Sun, Xian
# Writen by YuanZhiqiang, 2021.  Our code is depended on AMFMN
# ------------------------------------------------------------

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
import numpy as np
import yaml
import argparse
import utils
from vocab import deserialize_vocab
from PIL import Image

class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    """

    def __init__(self, data_split, vocab, opt):
        self.vocab = vocab
        self.loc = opt['dataset']['data_path']
        self.img_path = opt['dataset']['image_path']

        # Captions
        self.captions = []
        self.maxlength = 0

        # local features
        local_features = utils.load_from_npy(opt['dataset']['local_path'])[()]

        if data_split != 'test':
            with open(self.loc+'%s_caps_verify.txt' % data_split, 'rb') as f:
                for line in f:
                    self.captions.append(line.strip())

            self.images = []
            self.local_adj = []
            self.local_rep = []
            with open(self.loc + '%s_filename_verify.txt' % data_split, 'rb') as f:
                for line in f:
                    # local append
                    filename = str(line.strip())[2:-1].split(".")[0] + ".txt"
                    self.local_adj.append(np.array(local_features['adj_matrix'][filename]))
                    self.local_rep.append(np.array(local_features['local_rep'][filename]))

                    self.images.append(line.strip())
        else:
            with open(self.loc + '%s_caps.txt' % data_split, 'rb') as f:
                for line in f:
                    self.captions.append(line.strip())

            self.images = []
            self.local_adj = []
            self.local_rep = []
            with open(self.loc + '%s_filename.txt' % data_split, 'rb') as f:
                for line in f:
                    # local append
                    filename = str(line.strip())[2:-1].split(".")[0] + ".txt"
                    self.local_adj.append(np.array(local_features['adj_matrix'][filename]))
                    self.local_rep.append(np.array(local_features['local_rep'][filename]))

                    self.images.append(line.strip())

        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if len(self.images) != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

        if data_split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((278, 278)),
                transforms.RandomRotation(degrees=(0, 90)),
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index//self.im_div
        caption = self.captions[index]

        vocab = self.vocab

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            caption.lower().decode('utf-8'))
        punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
        tokens = [k for k in tokens if k not in punctuations]
        tokens_UNK = [k if k in vocab.word2idx.keys() else '<unk>' for k in tokens]


        caption = []
        caption.extend([vocab(token) for token in tokens_UNK])
        caption = torch.LongTensor(caption)

        image = Image.open(self.img_path  +str(self.images[img_id])[2:-1]).convert('RGB')
        image = self.transform(image)  # torch.Size([3, 256, 256])

        # local
        local_rep =  torch.from_numpy(self.local_rep[img_id]).type(torch.float32)
        local_adj = torch.from_numpy(self.local_adj[img_id]).type(torch.float32)


        return image, local_rep, local_adj, caption, tokens_UNK, index, img_id

    def __len__(self):
        return self.length


def collate_fn(data):

    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[4]), reverse=True)
    images, local_rep, local_adj, captions, tokens, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    local_rep = torch.stack(local_rep, 0)
    local_adj = torch.stack(local_adj, 0) # @@

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    lengths = [l if l !=0 else 1 for l in lengths]

    return images, local_rep, local_adj, targets, lengths, ids


def get_precomp_loader(data_split, vocab, batch_size=100,
                       shuffle=True, num_workers=0, opt={}):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_split, vocab, opt)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=False,
                                              collate_fn=collate_fn,
                                              num_workers=num_workers)
    return data_loader

def get_loaders(vocab, opt):
    train_loader = get_precomp_loader( 'train', vocab,
                                      opt['dataset']['batch_size'], True, opt['dataset']['workers'], opt=opt)
    val_loader = get_precomp_loader( 'val', vocab,
                                    opt['dataset']['batch_size_val'], False, opt['dataset']['workers'], opt=opt)
    return train_loader, val_loader


def get_test_loader(vocab, opt):
    test_loader = get_precomp_loader( 'test', vocab,
                                      opt['dataset']['batch_size_val'], False, opt['dataset']['workers'], opt=opt)
    return test_loader


# ====================================================================================================================================================
# ====================================================================================================================================================
# 加入名词进入原本的dataloader
def extract_nouns_with_position(text):
    # 对文本进行分词，并进行词性标注
    tagged_words = nltk.pos_tag(nltk.word_tokenize(text))

    # 存储名词和对应的位置信息
    noun_positions = []

    pos = 0
    for word, tag in tagged_words:
        # 名词通常以N开头的词性标签标记
        # 这包括 'NN' (名词), 'NNS' (复数名词), 'NNP' (专有名词), 'NNPS' (复数专有名词)
        if tag.startswith('N'):
            start_pos = text.find(word, pos)
            end_pos = start_pos + len(word)
            noun_positions.append((word, start_pos, end_pos))
            pos = end_pos

    return noun_positions


class PrecompDatasetNouns(data.Dataset):
    """
    Load precomputed captions and image features
    """

    def __init__(self, data_split, vocab, opt):
        self.vocab = vocab
        self.loc = opt['dataset']['data_path']
        self.img_path = opt['dataset']['image_path']

        # Captions
        self.captions = []
        self.maxlength = 0

        # local features
        local_features = utils.load_from_npy(opt['dataset']['local_path'])[()]

        if data_split != 'test':
            with open(self.loc+'%s_caps_verify.txt' % data_split, 'rb') as f:
                for line in f:
                    self.captions.append(line.strip())

            self.images = []
            self.local_adj = []
            self.local_rep = []
            with open(self.loc + '%s_filename_verify.txt' % data_split, 'rb') as f:
                for line in f:
                    # local append
                    filename = str(line.strip())[2:-1].split(".")[0] + ".txt"
                    self.local_adj.append(np.array(local_features['adj_matrix'][filename]))
                    self.local_rep.append(np.array(local_features['local_rep'][filename]))

                    self.images.append(line.strip())
        else:
            with open(self.loc + '%s_caps.txt' % data_split, 'rb') as f:
                for line in f:
                    self.captions.append(line.strip())

            self.images = []
            self.local_adj = []
            self.local_rep = []
            with open(self.loc + '%s_filename.txt' % data_split, 'rb') as f:
                for line in f:
                    # local append
                    filename = str(line.strip())[2:-1].split(".")[0] + ".txt"
                    self.local_adj.append(np.array(local_features['adj_matrix'][filename]))
                    self.local_rep.append(np.array(local_features['local_rep'][filename]))

                    self.images.append(line.strip())

        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if len(self.images) != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

        if data_split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((278, 278)),
                transforms.RandomRotation(degrees=(0, 90)),
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index//self.im_div
        caption = self.captions[index]

        vocab = self.vocab
        # locate the nouns in the caption
        noun_positions = extract_nouns_with_position(caption.lower().decode('utf-8'))
        noun_tokens = [noun_pos[0] for noun_pos in noun_positions]
        noun_positions_ids = [noun_pos[1] for noun_pos in noun_positions]
        noun_positions_ids.append(len(caption))
        
        
        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            caption.lower().decode('utf-8'))
        punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
        tokens = [k for k in tokens if k not in punctuations]
        tokens_UNK = [k if k in vocab.word2idx.keys() else '<unk>' for k in tokens]


        caption = []
        caption.extend([vocab(token) for token in tokens_UNK])
        caption = torch.LongTensor(caption)

        image = Image.open(self.img_path  +str(self.images[img_id])[2:-1]).convert('RGB')
        image = self.transform(image)  # torch.Size([3, 256, 256])

        # local
        local_rep =  torch.from_numpy(self.local_rep[img_id]).type(torch.float32)
        local_adj = torch.from_numpy(self.local_adj[img_id]).type(torch.float32)


        return image, local_rep, local_adj, caption, tokens_UNK, index, img_id

    def __len__(self):
        return self.length


def collate_fn_Nouns(data):

    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[4]), reverse=True)
    images, local_rep, local_adj, captions, tokens, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    local_rep = torch.stack(local_rep, 0)
    local_adj = torch.stack(local_adj, 0) # @@

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    lengths = [l if l !=0 else 1 for l in lengths]

    return images, local_rep, local_adj, targets, lengths, ids


def get_precomp_loader_Nouns(data_split, vocab, batch_size=100,
                       shuffle=True, num_workers=0, opt={}):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_split, vocab, opt)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=False,
                                              collate_fn=collate_fn,
                                              num_workers=num_workers)
    return data_loader

def get_loaders_Nouns(vocab, opt):
    train_loader = get_precomp_loader( 'train', vocab,
                                      opt['dataset']['batch_size'], True, opt['dataset']['workers'], opt=opt)
    val_loader = get_precomp_loader( 'val', vocab,
                                    opt['dataset']['batch_size_val'], False, opt['dataset']['workers'], opt=opt)
    return train_loader, val_loader


def get_test_loader_Nouns(vocab, opt):
    test_loader = get_precomp_loader( 'test', vocab,
                                      opt['dataset']['batch_size_val'], False, opt['dataset']['workers'], opt=opt)
    return test_loader

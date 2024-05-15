# -----------------------------------------------------------
# "Remote Sensing Cross-Modal Text-Image Retrieval Based on Global and Local Information"
# Yuan, Zhiqiang and Zhang, Wenkai and Changyuan Tian and Xuee, Rong and Zhengyuan Zhang and Wang, Hongqi and Fu, Kun and Sun, Xian
# Writen by YuanZhiqiang, 2021.  Our code is depended on AMFMN
# ------------------------------------------------------------

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import re
import nltk
import numpy as np
import yaml
import argparse
import utils
from vocab import deserialize_vocab
from PIL import Image
from transformers import BertTokenizer

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

# ==========================================================================
# ==========================================================================
# 加入bert的dataloader

class PrecompDatasetBert(data.Dataset):
    """
    Load precomputed captions and image features
    """

    def __init__(self, data_split, vocab, opt):
        self.vocab = vocab
        self.loc = opt['dataset']['data_path']
        self.img_path = opt['dataset']['image_path']
        self.bert_len = opt['model']['bert']['max_length']

        # Captions
        self.captions = [] # caption lines
        self.maxlength = 0
        
        ## @@ Bert config
        self.tokenizer = BertTokenizer.from_pretrained(opt['model']['bert']['bert_dir'])

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
        caption = self.captions[index] # caption in line
        bert_tokens = []
        vocab = self.vocab

        # Convert caption (string) to bert token ids.
        # print("caption split:",caption.decode('utf-8'))
        # print("caption re:",self.preprocess_text(caption.decode('utf-8')))
        
        for i, word in enumerate(self.preprocess_text(caption.decode('utf-8'))):
            btoken = self.tokenizer.tokenize(word)
            bert_tokens.extend(btoken)
        if len(bert_tokens) >= self.bert_len - 1:
            bert_tokens = bert_tokens[0:(self.bert_len - 2)]
        # print(bert_tokens)
            
        encode_dict = self.tokenizer.encode_plus(bert_tokens, max_length=self.bert_len, truncation=True, padding='max_length')
        input_ids, token_type_ids, attention_mask = encode_dict['input_ids'], encode_dict['token_type_ids'], \
                                                    encode_dict['attention_mask']
        input_ids=torch.LongTensor(input_ids)
        token_type_ids=torch.LongTensor(token_type_ids) 
        attention_mask=torch.LongTensor(attention_mask)
        
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

        return image, local_rep, local_adj, caption, tokens_UNK, index, img_id, input_ids, token_type_ids, attention_mask

    def __len__(self):
        return self.length
    
    # def preprocess_text(self, text):
    #     # 使用正则表达式将标点符号与单词分开
    #     text_out = re.sub(r'([,.!?()])', r' \1 ', text)
    #     text_out = re.sub(r'\s{2,}', ' ', text_out)  # 去除多余的空格
    #     return text_out.strip()
    # def preprocess_text(self, text):
    #     # 移除指定的符号
    #     text = re.sub(r'[.,?\"\'\']', '', text)
    #     return text.strip()
    def preprocess_text(self, text):

        punctuation = r"[!#$%&'()*+,-./:;<=>?@[\]^_`{|}~]"
        # 使用re.sub替换上述定义的标点符号为空字符，即删除这些符号
        cleaned_sentence = re.sub(punctuation, '', text)
        cleaned_text=cleaned_sentence.split(" ")
        return cleaned_text

def collate_fn_bert(data):

    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[4]), reverse=True)


    images, local_rep, local_adj, captions, tokens, ids, img_ids, \
        input_ids, token_type_ids, attention_mask = zip(*data)

    # @@ Merge input_ids, token_type_ids, attention_mask
    input_ids = torch.stack(input_ids, 0)
    token_type_ids = torch.stack(token_type_ids, 0)
    attention_mask = torch.stack(attention_mask, 0)
    
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

    return images, local_rep, local_adj, targets, lengths, ids, input_ids, token_type_ids, attention_mask


def get_precomp_loader_bert(data_split, vocab, batch_size=100,
                       shuffle=True, num_workers=0, opt={}):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDatasetBert(data_split, vocab, opt)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=False,
                                              collate_fn=collate_fn_bert,
                                              num_workers=num_workers)
    return data_loader

def get_loaders_bert(vocab, opt):
    train_loader = get_precomp_loader_bert( 'train', vocab,
                                      opt['dataset']['batch_size'], True, opt['dataset']['workers'], opt=opt)
    val_loader = get_precomp_loader( 'val', vocab,
                                    opt['dataset']['batch_size_val'], False, opt['dataset']['workers'], opt=opt)
    return train_loader, val_loader


def get_test_loader_bert(vocab, opt):
    test_loader = get_precomp_loader_bert( 'test', vocab, 
                                      opt['dataset']['batch_size_val'], False, opt['dataset']['workers'], opt=opt)
    return test_loader


if __name__ == '__main__':
    # Test dataloader
    from my_vocab import deserialize_vocab
    import argparse
    from torch.autograd import Variable


    def parser_options():
        # Hyper Parameters setting
        parser = argparse.ArgumentParser()
        parser.add_argument('--path_opt', default='option/RSITMD_mca/RSITMD_GaLR_my.yaml', type=str,
                            help='path to a yaml options file')
        # parser.add_argument('--text_sim_path', default='data/ucm_precomp/train_caps.npy', type=str,help='path to t2t sim matrix')
        opt = parser.parse_args()

        # load model options
        with open(opt.path_opt, 'r') as handle:
            options = yaml.safe_load(handle)

        return options

    options = parser_options()
    # print(options)
    vocab = deserialize_vocab(options['dataset']['vocab_path'])
    vocab_word = sorted(vocab.word2idx.items(), key=lambda x: x[1], reverse=False)
    vocab_word = [tup[0] for tup in vocab_word]

    # 测试原先的dataloader
    train_loader,_=get_loaders(vocab, options)
    for i, train_data in enumerate(train_loader):
        images, local_rep, local_adj, captions, lengths, ids= train_data
        batch_size = images.size(0)
        # measure data loading time


        input_text = Variable(captions)
        # print(input_text)
        print(input_text.type())
        print(input_text.size())
        if i>10:
            break
    
    # 测试bert的dataloader
    train_loader_bert,_=get_loaders_bert(vocab, options)
    
    for i, train_data in enumerate(train_loader_bert):
        images, local_rep, local_adj, captions, lengths, ids,\
        input_ids, token_type_ids, attention_mask= train_data
        batch_size = images.size(0)
        # measure data loading time

        # print(input_ids)
        input_text = Variable(input_ids)
        print(type(input_ids))    
        print(input_ids.size())
        break
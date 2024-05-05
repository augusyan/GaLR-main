# -----------------------------------------------------------
# "Remote Sensing Cross-Modal Text-Image Retrieval Based on Global and Local Information"
# Yuan, Zhiqiang and Zhang, Wenkai and Changyuan Tian and Xuee, Rong and Zhengyuan Zhang and Wang, Hongqi and Fu, Kun and Sun, Xian
# Writen by YuanZhiqiang, 2021.  Our code is depended on AMFMN
# ------------------------------------------------------------

import nltk
from collections import Counter
import argparse
import os
import json

annotations = {
    'coco_splits': ['train_caps.txt', 'val_caps.txt', 'test_caps.txt'],
    'flickr30k_splits': ['train_caps.txt', 'val_caps.txt', 'test_caps.txt'],
    'rsicd_precomp': ['train_caps.txt', 'val_caps.txt', 'test_caps.txt'],
    'rsitmd_precomp': ['train_caps.txt', 'val_caps.txt'],
    'ucm_precomp': ['train_caps.txt', 'val_caps.txt'],
    'sydney_precomp': ['train_caps.txt', 'val_caps.txt'],

    }


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def add_word(self, word):
        """        
        向词汇表中添加一个新词。
        参数:
        word: 要添加的词，字符串类型。

        返回值:
        无。
        """
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def serialize_vocab(vocab, dest):
    d = {}
    d['word2idx'] = vocab.word2idx
    d['idx2word'] = vocab.idx2word
    d['idx'] = vocab.idx
    with open(dest, "w") as f:
        json.dump(d, f)


def deserialize_vocab(src):
    """
    反序列化词汇表

    参数:
    src: 字符串，表示词汇表序列化文件的路径

    返回值:
    vocab: Vocabulary 类的实例，包含反序列化后的词汇表信息
    """
    with open(src) as f:  # 打开词汇表序列化文件
        d = json.load(f)  # 加载词汇表数据
    vocab = Vocabulary()  # 创建空的词汇表实例
    # 将序列化文件中的数据赋值给词汇表实例的属性
    vocab.word2idx = d['word2idx']
    vocab.idx2word = d['idx2word']
    vocab.idx = d['idx']
    return vocab

# 获得 captions
def from_txt(txt):
    captions = []
    with open(txt, 'rb') as f:
        for line in f:
            captions.append(line.strip())
    return captions


def build_vocab(data_path, data_name, caption_file, threshold):
    """
    构建一个简单的词汇表包装器。

    参数:
    - data_path: 数据路径，字符串。
    - data_name: 数据集名称，字符串。
    - caption_file: 包含图像标题文件路径的字典。
    - threshold: 词频阈值，仅包含出现次数大于等于此阈值的单词。

    返回:
    - vocab: 词汇表对象，包含所有词汇和特殊标记。
    """

    # 加载英语停用词列表
    stopword_list = list(set(nltk.corpus.stopwords.words('english')))
    counter = Counter()
    
    # 遍历标题文件，对每个文件中的标题进行分词和计数
    for path in caption_file[data_name]:
        full_path = os.path.join(os.path.join(data_path, data_name), path)
        captions = from_txt(full_path)  # 从文本文件中加载标题

        for i, caption in enumerate(captions):
            # 将标题转换为小写，分词，并移除标点符号和停用词
            tokens = nltk.tokenize.word_tokenize(
                caption.lower().decode('utf-8'))
            punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
            tokens = [k for k in tokens if k not in punctuations]
            tokens = [k for k in tokens if k not in stopword_list]
            counter.update(tokens)

            # 每处理1000个标题打印一次进度
            if i % 1000 == 0:
                print("[%d/%d] tokenized the captions." % (i, len(captions)))

    # 移除出现次数小于阈值的单词
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # 创建词汇表对象，并添加特殊标记
    vocab = Vocabulary()

    # 向词汇表中添加单词
    for i, word in enumerate(words):
        vocab.add_word(word)
    vocab.add_word('<unk>')  # 添加未知词汇标记

    return vocab


def main(data_path, data_name):
    vocab = build_vocab(data_path, data_name, caption_file=annotations, threshold=5)
    serialize_vocab(vocab, 'vocab/%s_vocab.json' % data_name)
    print("Saved vocabulary file to ", 'vocab/%s_vocab.json' %(data_name))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data')
    parser.add_argument('--data_name', default='sydney_precomp',
                        help='{coco,f30k}')
    opt = parser.parse_args()
    main(opt.data_path, opt.data_name)

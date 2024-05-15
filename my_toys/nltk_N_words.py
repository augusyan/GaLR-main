"""
    功能描述：
        使用nltk库，对文本进行分词、词性标注、提取名词等操作。
        将名词的位置信息提取出来，并入数据集，后续用作单独的Attention后者mask机制的输入。
    TODO:
        1. nltk提取名词
        2. 获得每个名词以及对应的起始、终止position信息 
    
"""
import nltk

# # 下载必要的语料库
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

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


# 文本数据
text = "There are two white houses next to the baseball field, which is located in the middle of the green lawn."

# 将文本分割成句子列表
sentences = nltk.sent_tokenize(text)

# 创建一个空列表来存储所有名词
all_nouns = []

# 遍历每个句子
for sentence in sentences:
    # 将句子分割成单词列表
    words = nltk.word_tokenize(sentence)
    
    # 为每个单词标注词性
    tagged_words = nltk.pos_tag(words)
    
    # 提取名词
    nouns = [word for word, pos in tagged_words if pos.startswith('NN')]
    
    # 将名词添加到结果列表中
    all_nouns.extend(nouns)

# 打印结果
print(all_nouns)
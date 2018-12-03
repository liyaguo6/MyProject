import numpy as np
import jieba
import re
import pandas as pd
import logging

jieba.setLogLevel(logging.INFO)


def load_data_files(data_file):
    """
    数据预处理
    :param data_file: filename
    :return: [['怎么样 治癫病 好','你 多大 了',…],[[0,1],[1,0],…]]
    """
    # Load data from files
    lines_0, lines_1, lines = read_and_clean_zh_file(data_file)
    # Generate labels
    lines_1 = [[0, 1] for _ in lines_1]
    lines_0 = [[1, 0] for _ in lines_0]
    y = np.concatenate([lines_0, lines_1], 0)
    return [lines, y]


def read_and_clean_zh_file(input_file, output_cleaned_file=None):
    df = pd.read_csv(input_file, header=None, encoding='gbk', names=['question', 'labels']).dropna()
    lines_0_raw = (list(df[df['labels'] == 0]['question']))
    lines_1_raw = (list(df[df['labels'] == 1]['question']))
    lines_0 = [clean_str(seperate_line(line)) for line in lines_0_raw]
    lines_1 = [clean_str(seperate_line(line)) for line in lines_1_raw]
    lines = lines_0 + lines_1
    if output_cleaned_file is not None:
        with open(output_cleaned_file, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write((line + '\n'))
    return lines_0, lines_1, lines


def clean_str(string):
    string = re.sub(r"[^\u4e00-\u9fff]", " ", string)
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # string = re.sub(r"\'s", " \'s", string)
    # string = re.sub(r"\'ve", " \'ve", string)
    # string = re.sub(r"n\'t", " n\'t", string)
    # string = re.sub(r"\'re", " \'re", string)
    # string = re.sub(r"\'d", " \'d", string)
    # string = re.sub(r"\'ll", " \'ll", string)
    # string = re.sub(r",", " , ", string)
    # string = re.sub(r"!", " ! ", string)
    # string = re.sub(r"\(", " \( ", string)
    # string = re.sub(r"\)", " \) ", string)
    # string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    # return string.strip().lower()
    return string.strip()


def seperate_line(line):
    return ' '.join(list(jieba.cut(line, cut_all=False)))


def padding_sentences(input_sentences, padding_token, padding_sentence_length=None):
    """
    1 计算句子最长的长度
    2 设置padding
    """
    sentences = [sentence.split(' ') for sentence in input_sentences]
    max_sentence_length = padding_sentence_length if padding_sentence_length is not None else max(
        [len(sentence) for sentence in sentences])
    sentences_list=[]
    for sentence in sentences:
        if len(sentence) > max_sentence_length:
            sentence = sentence[:max_sentence_length]
        else:
            sentence.extend([padding_token] * (max_sentence_length - len(sentence)))
        sentences_list.append(sentence)
    return (sentences_list, max_sentence_length)


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    '''
    Generate a batch iterator for a dataset
	1 计算样本数据的长度
	2 计算每一个epoch中batch的次数
	3 数据进行shuffle操作
	4 设置生成器
    '''
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            # Shuffle the data at each epoch
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_idx: end_idx]

import gensim
import jieba
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.functional import F


class RNNDataset(Dataset):
    def __init__(self, data, word2vec_model, seq_len, word_len=100):
        label = data['label'].to_numpy().astype(np.float32)
        self.word2vec_model = word2vec_model
        self.data = [list(jieba.cut(sentence)) for sentence in data['review'].to_list()]
        self.label = torch.from_numpy(label).unsqueeze(-1)
        self.seq_len = seq_len
        self.word_len = word_len        

    def preprocess(self, sentence):
        sentence = np.array([self.word2vec_model.wv[word] for word in sentence], dtype=np.float32)
        sentence = torch.from_numpy(sentence)
        sentence = F.pad(sentence, (0, 0, 0, self.seq_len - sentence.shape[0]), 'constant', 0)
        return sentence

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data[idx]
        label = self.label[idx]
        return self.preprocess(sentence), label
    

def build_word2vec_model(data, word_len):
    sentences = [list(jieba.cut(sentence)) for sentence in data['review'].to_list()]  # 分词
    if os.path.exists('./model/word2vec.model'):
        word2vec_model = gensim.models.Word2Vec.load('./model/word2vec.model')
    else:
        word2vec_model = gensim.models.Word2Vec(sentences, vector_size=word_len, window=5, min_count=1, workers=4)
        word2vec_model.save('./model/word2vec.model')
    return word2vec_model


def get_dataloader(data, seq_len, word_len, batch_size):
    data = data.dropna(axis=0, how='any')  # 删除有缺失值的行
    word2vec_model = build_word2vec_model(data, word_len)  # 创建word2vec模型
    
    # 打乱数据
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    data = data.iloc[idx]
    
    train_test_split = int(data.shape[0] * 0.8)
    train_data = data.iloc[:train_test_split]
    test_data = data.iloc[train_test_split:]
    train_dataset = RNNDataset(train_data, word2vec_model, seq_len, word_len)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataset = RNNDataset(test_data, word2vec_model, seq_len, word_len)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_dataloader, test_dataloader
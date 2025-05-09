# =========================================
# RNN Lab Tutorial
# 
# This script is exported from tutorial.ipynb
# =========================================

# =========================================
# Jupyter 简介 & 使用部分 (原Markdown单元格)
# =========================================

# =========================================
# RNN EXP 部分 - 自回归任务
# =========================================

import torch
from torch import nn
import numpy as np
vocab_size=32
# 定义句子
text = ['hey how are you','good i am fine','have a nice day']

# 连接所有句子并提取唯一字符
chars = set(''.join(text))

# 创建映射整数到字符的字典
int2char = dict(enumerate(chars))

# 创建映射字符到整数的字典
char2int = {char: ind for ind, char in int2char.items()}

print(char2int)
print(int2char)

# 找出最长句子的长度
maxlen = len(max(text, key=len))
print("The longest string has {} characters".format(maxlen))

# 填充
# 循环遍历句子列表，添加空格直到句子长度与最长句子相匹配
for i in range(len(text)):
    while len(text[i])<maxlen:
        text[i] += ' '

# 创建列表来存储输入和目标序列
input_seq = []
target_seq = []

for i in range(len(text)):
    # 移除最后一个字符得到输入序列
    input_seq.append(text[i][:-1])
    
    # 移除第一个字符得到目标序列
    target_seq.append(text[i][1:])
    print("Input Sequence: {}\nTarget Sequence: {}".format(input_seq[i], target_seq[i]))

# 将字符序列转换为整数序列
for i in range(len(text)):
    input_seq[i] = [char2int[character] for character in input_seq[i]]
    target_seq[i] = [char2int[character] for character in target_seq[i]]

# 定义关键变量
dict_size = len(char2int)
seq_len = maxlen - 1
batch_size = len(text)

def one_hot_encode(sequence, dict_size, seq_len, batch_size):
    # 创建指定输出形状的多维零数组
    features = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)
    
    # 在相关字符索引处将0替换为1以表示该字符
    for i in range(batch_size):
        for u in range(seq_len):
            features[i, u, sequence[i][u]] = 1
    return features

# 对输入序列进行one-hot编码
input_seq = one_hot_encode(input_seq, dict_size, seq_len, batch_size)
print("Input shape: {} --> (Batch Size, Sequence Length, One-Hot Encoding Size)".format(input_seq.shape))

# 将数据从numpy数组转换为PyTorch张量
input_seq = torch.from_numpy(input_seq)
target_seq = torch.Tensor(target_seq)

# 检查是否可以使用GPU
is_cuda = torch.cuda.is_available()

# 如果有GPU，将设备设置为GPU；否则使用CPU
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

# 定义模型类
class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        # 定义一些参数
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # 定义层
        # RNN层
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)

        # 使用下面定义的方法初始化第一个输入的隐藏状态
        hidden = self.init_hidden(batch_size)

        # 将输入和隐藏状态传入模型并获得输出
        out, hidden = self.rnn(x, hidden)
        
        # 重塑输出使其可以适合全连接层
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # 此方法生成我们将在前向传播中使用的第一个零隐藏状态
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        # 我们还将存放隐藏状态的张量发送到之前指定的设备
        return hidden

# 用超参数实例化模型
model = Model(input_size=dict_size, output_size=dict_size, hidden_dim=12, n_layers=1)
# 将模型设置到之前定义的设备上（默认是CPU）
model = model.to(device)

# 定义超参数
n_epochs = 100
lr=0.01

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 训练模型
input_seq = input_seq.to(device)
for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad() # 清除上一轮的梯度
    output, hidden = model(input_seq)
    output = output.to(device)
    target_seq = target_seq.to(device)
    loss = criterion(output, target_seq.view(-1).long())
    loss.backward() # 执行反向传播和计算梯度
    optimizer.step() # 相应地更新权重
    
    if epoch%10 == 0:
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))

# 定义预测辅助函数
@torch.no_grad()
def predict(model, character):
    # 对我们的输入进行one-hot编码以适应模型
    character = np.array([[char2int[c] for c in character]])
    character = one_hot_encode(character, dict_size, character.shape[1], 1)
    character = torch.from_numpy(character)
    character = character.to(device)
    
    out, hidden = model(character)

    prob = nn.functional.softmax(out[-1], dim=0).data
    # 从输出中取概率得分最高的类别
    char_ind = torch.max(prob, dim=0)[1].item()

    return int2char[char_ind], hidden

def sample(model, out_len, start='hey'):
    model.eval() # 评估模式
    start = start.lower()
    # 首先，遍历起始字符
    chars = [ch for ch in start]
    size = out_len - len(chars)
    # 现在传入前面的字符并获取新字符
    for ii in range(size):
        char, h = predict(model, chars)
        chars.append(char)

    return ''.join(chars)

# 测试模型
print(sample(model, 15, 'good'))

# =========================================
# 中文影评情感分类任务
# =========================================

# 设置随机种子以保证实验的可重复性
import random
import numpy as np
import torch

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("随机种子已设置为", SEED)

# 导入所需库
import gensim
import jieba
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.functional import F
import pandas as pd 

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Matplotlib 中文显示设置
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题


class RNNDataset(Dataset):
    def __init__(self, data, word2vec_model, seq_len, word_len=100):
        # 假设'data'是DataFrame
        label = data['label'].to_numpy().astype(np.float32)
        self.word2vec_model = word2vec_model
        # 确保评论是字符串类型，然后传递给jieba.cut
        self.data = [list(jieba.cut(str(sentence))) for sentence in data['review'].to_list()]
        self.label = torch.from_numpy(label).unsqueeze(-1)
        self.seq_len = seq_len
        self.word_len = word_len        

    def preprocess(self, sentence):
        # 过滤掉Word2Vec词汇表中不存在的单词并处理空句子
        valid_words = [word for word in sentence if word in self.word2vec_model.wv]
        if not valid_words:
            # 如果没有有效单词，则返回一个适当形状的零张量
            return torch.zeros((self.seq_len, self.word_len), dtype=torch.float32)

        sentence_vectors = np.array([self.word2vec_model.wv[word] for word in valid_words], dtype=np.float32)
        sentence_tensor = torch.from_numpy(sentence_vectors)
        
        # 填充或截断
        current_len = sentence_tensor.shape[0]
        if current_len == 0:  # 应该被空valid_words检查捕获，但作为保障
             return torch.zeros((self.seq_len, self.word_len), dtype=torch.float32)

        if current_len > self.seq_len:
            sentence_tensor = sentence_tensor[:self.seq_len, :]
        elif current_len < self.seq_len:
            padding_size = self.seq_len - current_len
            # 为(seq_len, word_len)形状正确填充
            sentence_tensor = F.pad(sentence_tensor, (0, 0, 0, padding_size), 'constant', 0)
        return sentence_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data[idx]
        label = self.label[idx]
        return self.preprocess(sentence), label
    

def build_word2vec_model(data, word_len, model_dir='./model', model_name='word2vec.model'):
    # 假设'data'是DataFrame
    # 确保评论是字符串类型，然后传递给jieba.cut
    sentences = [list(jieba.cut(str(sentence))) for sentence in data['review'].to_list()]
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, model_name)
        
    if os.path.exists(model_path):
        print(f"Loading existing Word2Vec model from {model_path}")
        word2vec_model = gensim.models.Word2Vec.load(model_path)
    else:
        print(f"Building new Word2Vec model and saving to {model_path}")
        word2vec_model = gensim.models.Word2Vec(sentences, vector_size=word_len, window=5, min_count=1, workers=4)
        word2vec_model.save(model_path)
    return word2vec_model


def get_dataloader(data_path, seq_len, word_len, batch_size, model_dir='./model'):
    # 从CSV加载数据，假设data_path是CSV文件的字符串路径
    try:
        data_df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error loading data from {data_path}: {e}")
        print("Using sample data instead.")
        # 如果加载失败，回退到示例数据
        sample_reviews = [
            '这家酒店环境很好，服务周到。', '房间太小了，性价比不高。', 
            '早餐种类丰富，味道不错。', '隔音效果差，晚上很吵。', 
            '地理位置优越，出行方便。', '设施陈旧，体验不佳。',
            '服务员态度热情，点赞。', '网络信号不稳定。',
            '强烈推荐这家！', '不会再来了。'
        ]
        sample_labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] # 1表示正面，0表示负面
        data_df = pd.DataFrame({'review': sample_reviews, 'label': sample_labels})

    # 确保'review'和'label'列存在
    if 'review' not in data_df.columns or 'label' not in data_df.columns:
        raise ValueError("DataFrame必须包含'review'和'label'列。")

    data_df = data_df.dropna(subset=['review', 'label'], axis=0, how='any')  # 删除有缺失值的行
    data_df['review'] = data_df['review'].astype(str) # 确保评论是字符串
    data_df['label'] = data_df['label'].astype(int) # 确保标签是整数

    if data_df.empty:
        raise ValueError("数据清洗后为空。请检查您的数据文件。")

    word2vec_model = build_word2vec_model(data_df, word_len, model_dir=model_dir)  # 创建word2vec模型
    
    # 打乱数据
    idx = np.arange(data_df.shape[0])
    np.random.shuffle(idx)
    data_df_shuffled = data_df.iloc[idx]
    
    split_ratio = 0.8
    train_test_split_index = int(data_df_shuffled.shape[0] * split_ratio)

    if train_test_split_index == 0 or train_test_split_index == data_df_shuffled.shape[0]:
        raise ValueError(f"训练/测试分割产生了一个空集合。总样本数：{data_df_shuffled.shape[0]}，分割索引：{train_test_split_index}。请调整数据或分割比例。")
    
    train_data = data_df_shuffled.iloc[:train_test_split_index]
    test_data = data_df_shuffled.iloc[train_test_split_index:]
    
    train_dataset = RNNDataset(train_data, word2vec_model, seq_len, word_len)
    # 如果出现问题，考虑num_workers=0，特别是在Windows或某些Jupyter环境中
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) 
    
    test_dataset = RNNDataset(test_data, word2vec_model, seq_len, word_len)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Data loaded: {len(data_df)} total reviews.")
    print(f"Train Dataloader: {len(train_dataloader.dataset)} samples, {len(train_dataloader)} batches.")
    print(f"Test Dataloader: {len(test_dataloader.dataset)} samples, {len(test_dataloader)} batches.")
    
    # 如果notebook的其他部分需要，返回vocab相关信息
    # 现在，按照提示中的原始函数签名只返回dataloaders
    return train_dataloader, test_dataloader

# 导入更多必要的库
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split 
from torch.nn.utils.rnn import pad_sequence
from collections import Counter # 用于构建词汇表
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# 数据加载
# 定义参数
DATA_FILE_PATH = "data/ChnSentiCorp_htl_all.csv"  # 修改为您的数据文件路径
# DATA_FILE_PATH = "data/sample_reviews.csv" # 使用示例数据进行测试
SEQ_LEN = 50  # 序列长度
WORD_LEN = 100  # 词向量维度
BATCH_SIZE = 64  # 批处理大小
MODEL_DIR = './model_w2v' # Word2Vec 模型保存目录

# 创建一个示例CSV文件，如果真实数据文件不存在 (可选, 用于快速测试)
if not os.path.exists(DATA_FILE_PATH) and DATA_FILE_PATH != "data/sample_reviews.csv":
    print(f"警告: 数据文件 {DATA_FILE_PATH} 未找到。")

# 调用 get_dataloader
try:
    train_loader, test_loader = get_dataloader(DATA_FILE_PATH, SEQ_LEN, WORD_LEN, BATCH_SIZE, model_dir=MODEL_DIR)
    print(f"成功创建 DataLoader。")
    
    # 检查 DataLoader 是否有数据 (可选)
    if train_loader and len(train_loader.dataset) > 0:
        print(f"训练集样本数: {len(train_loader.dataset)}")
        # 获取一批数据以查看
        sample_batch_text, sample_batch_labels = next(iter(train_loader))
        print(f"一批训练数据的形状: {sample_batch_text.shape}")
        print(f"一批训练标签的形状: {sample_batch_labels.shape}")
    else:
        print("训练数据加载器为空或没有数据。")

    if test_loader and len(test_loader.dataset) > 0:
        print(f"测试集样本数: {len(test_loader.dataset)}")
    else:
        print("测试数据加载器为空或没有数据。")

except Exception as e:
    print(f"创建 DataLoader 时出错: {e}")
    print("请检查 get_dataloader 函数和数据文件路径。")

# 定义改进后的RNN模型
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout_prob, use_embedding=True, vocab_size=None, embedding_dim=None, pad_idx=None):
        super().__init__()
        self.use_embedding = use_embedding
        if self.use_embedding:
            if vocab_size is None or embedding_dim is None:
                raise ValueError("vocab_size and embedding_dim must be provided if use_embedding is True.")
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
            rnn_input_dim = embedding_dim
        else:
            rnn_input_dim = input_dim # This is the word_len from Word2Vec
            
        # 使用GRU代替简单RNN，提高模型性能
        self.rnn = nn.GRU(rnn_input_dim, 
                         hidden_dim, 
                         num_layers=n_layers, 
                         bidirectional=True, # 启用双向RNN以捕获更多上下文信息
                         batch_first=True,
                         dropout=dropout_prob if n_layers > 1 else 0)
                         
        # 添加额外的全连接层进行特征处理
        fc_input_dim = hidden_dim * 2  # 双向所以是2倍
        self.fc1 = nn.Linear(fc_input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)  # 批归一化加速训练

    def forward(self, text_input):
        if self.use_embedding:
            embedded = self.dropout(self.embedding(text_input))
        else:
            embedded = self.dropout(text_input)
        
        output, hidden = self.rnn(embedded)
        
        # 使用最后一个时间步的隐藏状态
        if self.rnn.bidirectional:
            # 连接正向和反向的最终隐藏状态
            final_hidden_state = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            final_hidden_state = hidden[-1,:,:]
            
        # 通过额外的全连接层
        dense1 = self.fc1(self.dropout(final_hidden_state))
        dense1 = self.batch_norm(dense1)
        dense1 = self.act(dense1)
        output = self.fc2(self.dropout(dense1))
        
        return output

# 定义改进后的LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout_prob, use_embedding=True, vocab_size=None, embedding_dim=None, pad_idx=None):
        super().__init__()
        self.use_embedding = use_embedding
        if self.use_embedding:
            if vocab_size is None or embedding_dim is None:
                raise ValueError("vocab_size and embedding_dim must be provided if use_embedding is True.")
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
            lstm_input_dim = embedding_dim
        else:
            lstm_input_dim = input_dim
        
        # 增加隐藏层维度和层数
        self.lstm = nn.LSTM(lstm_input_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           batch_first=True,
                           dropout=dropout_prob if n_layers > 1 else 0)
        
        # 添加注意力机制
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = nn.Linear(fc_input_dim, 1)
        
        # 增加输出层复杂度
        self.fc1 = nn.Linear(fc_input_dim, fc_input_dim // 2)
        self.act = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(fc_input_dim // 2)
        self.fc2 = nn.Linear(fc_input_dim // 2, output_dim)
        self.dropout = nn.Dropout(dropout_prob)
        
    def attention_net(self, lstm_output):
        # lstm_output: [batch_size, seq_len, hidden_dim*2]
        attn_weights = F.softmax(self.attention(lstm_output), dim=1)
        # attn_weights: [batch_size, seq_len, 1]
        context = torch.sum(attn_weights * lstm_output, dim=1)
        # context: [batch_size, hidden_dim*2]
        return context
        
    def forward(self, text_input):
        if self.use_embedding:
            embedded = self.dropout(self.embedding(text_input))
        else:
            embedded = self.dropout(text_input)
        
        # lstm_output: [batch_size, seq_len, hidden_dim*2]
        # hidden/cell: [n_layers*2, batch_size, hidden_dim]
        lstm_output, (hidden, cell) = self.lstm(embedded)
        
        # 使用注意力机制
        attn_output = self.attention_net(lstm_output)
        
        # 通过深层全连接网络
        dense1 = self.fc1(self.dropout(attn_output))
        dense1 = self.batch_norm(dense1)
        dense1 = self.act(dense1)
        output = self.fc2(self.dropout(dense1))
        
        return output

# RNN模型超参数优化
HIDDEN_DIM_RNN = 256  # 增加隐藏层维度
OUTPUT_DIM = 1       
N_LAYERS_RNN = 2      # 增加层数
DROPOUT_RNN = 0.5     # 增加dropout以防止过拟合

# 使用预计算的Word2Vec向量，所以不需要嵌入层
rnn_model = RNNModel(input_dim=WORD_LEN, 
                     hidden_dim=HIDDEN_DIM_RNN, 
                     output_dim=OUTPUT_DIM, 
                     n_layers=N_LAYERS_RNN, 
                     dropout_prob=DROPOUT_RNN,
                     use_embedding=False,  # 修改为False，使用预计算的Word2Vec向量
                     vocab_size=None, 
                     embedding_dim=None, 
                     pad_idx=None).to(device)

# 添加权重衰减和梯度裁剪
LEARNING_RATE_RNN = 0.001
WEIGHT_DECAY_RNN = 1e-5  # L2正则化
optimizer_rnn = optim.Adam(rnn_model.parameters(), lr=LEARNING_RATE_RNN, weight_decay=WEIGHT_DECAY_RNN)

# 添加学习率调度器，移除不支持的verbose参数
scheduler_rnn = optim.lr_scheduler.ReduceLROnPlateau(optimizer_rnn, mode='min', factor=0.5, patience=3)

# LSTM模型超参数优化
HIDDEN_DIM_LSTM = 256  # 增加隐藏层维度
N_LAYERS_LSTM = 2      # 增加层数
BIDIRECTIONAL_LSTM = True 
DROPOUT_LSTM = 0.5     # 增加dropout以防止过拟合

# 使用预计算的Word2Vec向量，所以不需要嵌入层
lstm_model = LSTMModel(input_dim=WORD_LEN,
                       hidden_dim=HIDDEN_DIM_LSTM, 
                       output_dim=OUTPUT_DIM, 
                       n_layers=N_LAYERS_LSTM, 
                       bidirectional=BIDIRECTIONAL_LSTM, 
                       dropout_prob=DROPOUT_LSTM,
                       use_embedding=False,  # 修改为False，使用预计算的Word2Vec向量
                       vocab_size=None,
                       embedding_dim=None,
                       pad_idx=None).to(device)

# 添加权重衰减和梯度裁剪
LEARNING_RATE_LSTM = 0.001
WEIGHT_DECAY_LSTM = 1e-5  # L2正则化
optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE_LSTM, weight_decay=WEIGHT_DECAY_LSTM)

# 添加学习率调度器，移除不支持的verbose参数
scheduler_lstm = optim.lr_scheduler.ReduceLROnPlateau(optimizer_lstm, mode='min', factor=0.5, patience=3)

# 修改训练函数以添加学习率调度使用
def train_model(model, iterator, optimizer, criterion, scheduler=None, clip=1.0):
    epoch_loss = 0
    epoch_acc = 0 
    
    model.train()
    
    if not iterator or len(iterator) == 0:
        print("训练迭代器为空，跳过训练。")
        return 0, 0

    for batch_idx, (text_batch, labels_batch) in enumerate(iterator):
        text_batch = text_batch.to(device)
        labels_batch = labels_batch.to(device) 
        
        optimizer.zero_grad()
        
        predictions = model(text_batch) 
        
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            predictions = predictions.squeeze(-1) if predictions.dim() > 1 and predictions.size(-1) == 1 else predictions
            labels_batch = labels_batch.squeeze(-1).float() if labels_batch.dim() > 1 and labels_batch.size(-1) == 1 else labels_batch.float()
            loss = criterion(predictions, labels_batch)
            rounded_preds = torch.round(torch.sigmoid(predictions))
            correct = (rounded_preds == labels_batch).float()
            acc = correct.sum() / len(correct)
        elif isinstance(criterion, nn.CrossEntropyLoss):
            labels_batch = labels_batch.squeeze(-1).long() if labels_batch.dim() > 1 and labels_batch.size(-1) == 1 else labels_batch.long()
            loss = criterion(predictions, labels_batch)
            top_p, top_class = predictions.topk(1, dim=1)
            correct = (top_class.squeeze(1) == labels_batch).float()
            acc = correct.sum() / len(correct)
        else:
            raise ValueError("不支持的损失函数类型")

        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    # 如果提供了调度器，更新学习率
    if scheduler is not None:
        scheduler.step(epoch_loss / len(iterator))
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# 早停和模型保存
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='best_model.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
        
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'验证损失减少 ({self.val_loss_min:.6f} --> {val_loss:.6f})。保存模型...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
# 在训练RNN模型前创建早停实例
early_stopping_rnn = EarlyStopping(patience=5, verbose=True, path='best_rnn_model.pt')

# 在训练LSTM模型前创建早停实例
early_stopping_lstm = EarlyStopping(patience=5, verbose=True, path='best_lstm_model.pt')

# 定义评估函数
def evaluate_model(model, iterator, criterion):
    epoch_loss = 0
    # epoch_acc = 0 # Replaced by sklearn.metrics.accuracy_score for final accuracy
    
    all_preds_list = []
    all_labels_list = []
    
    model.eval()

    if not iterator or len(iterator) == 0:
        print("评估迭代器为空，跳过评估。")
        return 0, 0, np.array([]), [], []
    
    with torch.no_grad():
        for text_batch, labels_batch in iterator:
            text_batch = text_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            predictions = model(text_batch)
            
            current_labels_np = (labels_batch.squeeze(-1).cpu().numpy() if labels_batch.dim() > 1 and labels_batch.size(-1) == 1 else labels_batch.cpu().numpy())
            all_labels_list.extend(current_labels_np)

            if isinstance(criterion, nn.BCEWithLogitsLoss):
                predictions_squeezed = predictions.squeeze(-1) if predictions.dim() > 1 and predictions.size(-1) == 1 else predictions
                labels_batch_squeezed = labels_batch.squeeze(-1).float() if labels_batch.dim() > 1 and labels_batch.size(-1) == 1 else labels_batch.float()
                loss = criterion(predictions_squeezed, labels_batch_squeezed)
                rounded_preds = torch.round(torch.sigmoid(predictions_squeezed))
                all_preds_list.extend(rounded_preds.cpu().numpy())
            elif isinstance(criterion, nn.CrossEntropyLoss):
                labels_batch_squeezed = labels_batch.squeeze(-1).long() if labels_batch.dim() > 1 and labels_batch.size(-1) == 1 else labels_batch.long()
                loss = criterion(predictions, labels_batch_squeezed)
                top_p, top_class = predictions.topk(1, dim=1)
                all_preds_list.extend(top_class.squeeze(1).cpu().numpy())
            else:
                raise ValueError("不支持的损失函数类型")

            epoch_loss += loss.item()
            # Per-batch accuracy calculation removed, final accuracy calculated once at the end
            
    # Ensure all_labels_list and all_preds_list are not empty before calculating metrics
    if not all_labels_list or not all_preds_list:
        print("没有标签或预测结果可用于计算指标。")
        return epoch_loss / len(iterator) if len(iterator) > 0 else 0, 0, np.array([]), [], []

    # Convert lists to numpy arrays for sklearn metrics
    all_labels_np = np.array(all_labels_list)
    all_preds_np = np.array(all_preds_list)

    # Ensure labels are integer type for confusion_matrix and accuracy_score if they were float (e.g. from BCE)
    final_accuracy = accuracy_score(all_labels_np.astype(int), all_preds_np.astype(int))
    cm = confusion_matrix(all_labels_np.astype(int), all_preds_np.astype(int))
    
    return epoch_loss / len(iterator), final_accuracy, cm, all_labels_np, all_preds_np

# 定义可视化函数
def plot_loss_curves(train_losses, val_losses, title_prefix=""):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='训练损失 (Train Loss)')
    if val_losses: # 如果提供了验证损失
        plt.plot(val_losses, label='验证损失 (Validation Loss)')
    plt.title(f'{title_prefix} 训练和验证损失曲线')
    plt.xlabel('轮次 (Epochs)')
    plt.ylabel('损失 (Loss)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(cm, class_names, title_prefix=""):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{title_prefix} 混淆矩阵 (Confusion Matrix)')
    plt.xlabel('预测标签 (Predicted Label)')
    plt.ylabel('真实标签 (True Label)')
    plt.show()

# RNN模型实验
# 超参数设置
# 确定我们是否使用预计算的嵌入（从RNNDataset通过Word2Vec）
# 这个标志应该已经在数据加载/准备单元格中设置
if 'USING_PRECOMPUTED_EMBEDDINGS' not in globals():
    print("警告: USING_PRECOMPUTED_EMBEDDINGS 未定义。假设为 True。")
    USING_PRECOMPUTED_EMBEDDINGS = True 

if 'EMBEDDING_DIM_MODEL' not in globals():
    print("警告: EMBEDDING_DIM_MODEL 未定义。设置为默认值 100。")
    EMBEDDING_DIM_MODEL = 100 # 默认值如果未设置，应该是WORD_LEN

RNN_INPUT_DIM = EMBEDDING_DIM_MODEL if USING_PRECOMPUTED_EMBEDDINGS else vocab_size
RNN_USE_EMBEDDING_LAYER = False  # 明确设置为False

# 仅当RNN_USE_EMBEDDING_LAYER为True时才需要vocab_size和PAD_IDX
# 确保它们在需要时从前面的单元格定义。
if RNN_USE_EMBEDDING_LAYER:
    if 'PAD_IDX' not in globals(): PAD_IDX = 0 # 默认PAD_IDX如果未定义
    RNN_VOCAB_SIZE = vocab_size
    RNN_EMBEDDING_DIM_INTERNAL = EMBEDDING_DIM_MODEL # 如果使用，嵌入层的维度
    RNN_PAD_IDX = PAD_IDX
else: # 不使用内部嵌入层
    RNN_VOCAB_SIZE = None
    RNN_EMBEDDING_DIM_INTERNAL = None
    RNN_PAD_IDX = None


HIDDEN_DIM_RNN = 256 
OUTPUT_DIM = 1       
N_LAYERS_RNN = 2     
DROPOUT_RNN = 0.5    

rnn_model = RNNModel(input_dim=RNN_INPUT_DIM, # 如果没有嵌入，则为特征维度，否则为vocab_size
                     hidden_dim=HIDDEN_DIM_RNN, 
                     output_dim=OUTPUT_DIM, 
                     n_layers=N_LAYERS_RNN, 
                     dropout_prob=DROPOUT_RNN,
                     use_embedding=False, # 明确设置为False
                     vocab_size=None,
                     embedding_dim=None,
                     pad_idx=None
                    ).to(device)

LEARNING_RATE_RNN = 0.001
optimizer_rnn = optim.Adam(rnn_model.parameters(), lr=LEARNING_RATE_RNN)

if OUTPUT_DIM == 1:
    criterion_rnn = nn.BCEWithLogitsLoss().to(device) 
else: 
    criterion_rnn = nn.CrossEntropyLoss().to(device)

print("RNN 模型结构:")
print(rnn_model)

try:
    if len(train_loader) > 0:
        sample_batch_text, sample_batch_labels = next(iter(train_loader))
        sample_batch_text = sample_batch_text.to(device)
        with torch.no_grad():
            output = rnn_model(sample_batch_text)
        print(f"RNN 模型单批次输出形状: {output.shape}") 
    else:
        print("训练加载器为空，无法测试RNN模型批处理。")
except Exception as e:
    print(f"测试RNN模型时出错: {e}")

# 训练RNN模型
N_EPOCHS = 30 # 训练轮次，根据需要调整

rnn_train_losses = []
# rnn_val_losses = [] # 如果有验证集，可以记录验证损失

print("开始训练 RNN 模型...")
if not list(train_loader): # 检查 train_loader 是否为空
    print("错误：训练数据加载器为空。无法开始训练。请检查数据预处理步骤。")
else:
    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train_model(rnn_model, train_loader, optimizer_rnn, criterion_rnn, scheduler_rnn)
        # val_loss, val_acc, _, _, _ = evaluate_model(rnn_model, test_loader, criterion_rnn) # 假设test_loader作为验证集
        
        rnn_train_losses.append(train_loss)
        # rnn_val_losses.append(val_loss)
        
        print(f'RNN Epoch: {epoch+1:02}')
        print(f'\t训练损失: {train_loss:.3f} | 训练准确率: {train_acc*100:.2f}%')
        # print(f'\t验证损失: {val_loss:.3f} | 验证准确率: {val_acc*100:.2f}%')
    print("RNN 模型训练完成。")

# 绘制损失曲线
plot_loss_curves(rnn_train_losses, None, title_prefix="RNN") # 如果有验证损失，传入 rnn_val_losses

# 评估RNN模型
if not list(test_loader) or len(test_loader.dataset) == 0: # 检查数据集是否也非空
    print("错误：测试数据加载器为空或不包含数据。无法进行评估。")
    # 定义占位符，以便评估被跳过时防止后面出现NameError
    test_loss_rnn, test_acc_rnn, cm_rnn = 0.0, 0.0, np.array([])
elif 'rnn_model' not in locals() or ('rnn_train_losses' in locals() and not rnn_train_losses and N_EPOCHS > 0) : # 检查模型是否存在且已训练（如果epochs > 0）
    print("错误：RNN模型未定义或未训练/训练失败，无法评估。")
    test_loss_rnn, test_acc_rnn, cm_rnn = 0.0, 0.0, np.array([])
else:
    test_loss_rnn, test_acc_rnn, cm_rnn, _, _ = evaluate_model(rnn_model, test_loader, criterion_rnn)
    print(f'RNN 测试损失: {test_loss_rnn:.3f} | RNN 测试准确率: {test_acc_rnn*100:.2f}%')

    class_names = ['负面 (0)', '正面 (1)'] 
    if cm_rnn.size > 0: # 仅当混淆矩阵非空时绘图
        plot_confusion_matrix(cm_rnn, class_names, title_prefix="RNN")
    else:
        print("混淆矩阵为空，不绘制。")

# LSTM模型实验
# LSTM模型超参数

LSTM_INPUT_DIM = EMBEDDING_DIM_MODEL if USING_PRECOMPUTED_EMBEDDINGS else vocab_size
LSTM_USE_EMBEDDING_LAYER = False # 明确设置为False

if LSTM_USE_EMBEDDING_LAYER:
    if 'vocab_size' not in globals(): raise ValueError("vocab_size is needed for LSTM embedding layer but not defined.")
    if 'PAD_IDX' not in globals(): PAD_IDX = 0 
    LSTM_VOCAB_SIZE = vocab_size
    LSTM_EMBEDDING_DIM_INTERNAL = EMBEDDING_DIM_MODEL
    LSTM_PAD_IDX = PAD_IDX
else:
    LSTM_VOCAB_SIZE = None
    LSTM_EMBEDDING_DIM_INTERNAL = None
    LSTM_PAD_IDX = None


HIDDEN_DIM_LSTM = 256 
# OUTPUT_DIM与RNN相同
N_LAYERS_LSTM = 2     
BIDIRECTIONAL_LSTM = True 
DROPOUT_LSTM = 0.5    

lstm_model = LSTMModel(input_dim=LSTM_INPUT_DIM,
                       hidden_dim=HIDDEN_DIM_LSTM, 
                       output_dim=OUTPUT_DIM, 
                       n_layers=N_LAYERS_LSTM, 
                       bidirectional=BIDIRECTIONAL_LSTM, 
                       dropout_prob=DROPOUT_LSTM,
                       use_embedding=False, # 明确设置为False
                       vocab_size=None,
                       embedding_dim=None,
                       pad_idx=None
                      ).to(device)

LEARNING_RATE_LSTM = 0.001
optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE_LSTM)

if OUTPUT_DIM == 1:
    criterion_lstm = nn.BCEWithLogitsLoss().to(device)
else:
    criterion_lstm = nn.CrossEntropyLoss().to(device)

print("LSTM 模型结构:")
print(lstm_model)

try:
    if len(train_loader) > 0:
        sample_batch_text, sample_batch_labels = next(iter(train_loader))
        sample_batch_text = sample_batch_text.to(device)
        with torch.no_grad():
            output = lstm_model(sample_batch_text)
        print(f"LSTM 模型单批次输出形状: {output.shape}")
    else:
        print("训练加载器为空，无法测试LSTM模型批处理。")
except Exception as e:
    print(f"测试LSTM模型时出错: {e}")

# 训练LSTM模型
# N_EPOCHS保持一致，方便比较
lstm_train_losses = []
# lstm_val_losses = []

print("开始训练 LSTM 模型...")
if not list(train_loader):
    print("错误：训练数据加载器为空。无法开始训练。")
else:
    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train_model(lstm_model, train_loader, optimizer_lstm, criterion_lstm)
        # val_loss, val_acc, _, _, _ = evaluate_model(lstm_model, test_loader, criterion_lstm)
        
        lstm_train_losses.append(train_loss)
        # lstm_val_losses.append(val_loss)
        
        print(f'LSTM Epoch: {epoch+1:02}')
        print(f'\t训练损失: {train_loss:.3f} | 训练准确率: {train_acc*100:.2f}%')
        # print(f'\t验证损失: {val_loss:.3f} | 验证准确率: {val_acc*100:.2f}%')
    print("LSTM 模型训练完成。")

# 绘制损失曲线
plot_loss_curves(lstm_train_losses, None, title_prefix="LSTM")

# 评估LSTM模型
if not list(test_loader) or len(test_loader.dataset) == 0: # 检查数据集是否也非空
    print("错误：测试数据加载器为空或不包含数据。无法进行评估。")
    test_loss_lstm, test_acc_lstm, cm_lstm = 0.0, 0.0, np.array([])
elif 'lstm_model' not in locals() or ('lstm_train_losses' in locals() and not lstm_train_losses and N_EPOCHS > 0):
    print("错误：LSTM模型未定义或未训练/训练失败，无法评估。")
    test_loss_lstm, test_acc_lstm, cm_lstm = 0.0, 0.0, np.array([])
else:
    test_loss_lstm, test_acc_lstm, cm_lstm, _, _ = evaluate_model(lstm_model, test_loader, criterion_lstm)
    print(f'LSTM 测试损失: {test_loss_lstm:.3f} | LSTM 测试准确率: {test_acc_lstm*100:.2f}%')
    
    if cm_lstm.size > 0: # class_names在RNN评估单元格中定义
        plot_confusion_matrix(cm_lstm, class_names, title_prefix="LSTM")
    else:
        print("LSTM 混淆矩阵为空，不绘制。")

# 性能对比
print("性能对比:")
if 'test_acc_rnn' in locals() and 'test_acc_lstm' in locals():
    print(f"RNN 模型测试准确率: {test_acc_rnn*100:.2f}%")
    print(f"LSTM 模型测试准确率: {test_acc_lstm*100:.2f}%")

    if test_acc_lstm > test_acc_rnn:
        print("LSTM 模型在此任务和超参数设置下表现更好。")
    elif test_acc_rnn > test_acc_lstm:
        print("RNN 模型在此任务和超参数设置下表现更好。")
    else:
        print("RNN 和 LSTM 模型表现相当。")
else:
    print("未能完成两个模型的评估，无法进行比较。请检查之前的步骤。")

# 模型性能可视化对比
# 绘制RNN和LSTM的训练损失曲线对比
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
epochs = range(1, len(rnn_train_losses) + 1)
plt.plot(epochs, rnn_train_losses, 'b-', label='RNN')
plt.plot(epochs, lstm_train_losses, 'r-', label='LSTM')
plt.title('训练损失对比')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 绘制测试准确率对比 (柱状图)
plt.subplot(1, 2, 2)
model_names = ['RNN', 'LSTM']
accuracies = [test_acc_rnn * 100, test_acc_lstm * 100]
bars = plt.bar(model_names, accuracies, color=['blue', 'red'])
plt.title('测试准确率对比')
plt.ylabel('准确率 (%)')
plt.ylim([0, 100])  # 确保y轴从0到100

# 在柱状图上添加准确率数值标签
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.2f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 绘制混淆矩阵对比
if cm_rnn.size > 0 and cm_lstm.size > 0:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.heatmap(cm_rnn, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, 
                yticklabels=class_names, ax=ax1, cbar=False)
    ax1.set_title('RNN 混淆矩阵')
    ax1.set_xlabel('预测标签')
    ax1.set_ylabel('真实标签')
    
    sns.heatmap(cm_lstm, annot=True, fmt='d', cmap='Reds', xticklabels=class_names, 
                yticklabels=class_names, ax=ax2, cbar=False)
    ax2.set_title('LSTM 混淆矩阵')
    ax2.set_xlabel('预测标签')
    ax2.set_ylabel('真实标签')
    
    plt.tight_layout()
    plt.show()

# 模型预测示例分析
# 选择一些测试样本进行分析
def predict_samples(model, data_loader, num_samples=5):
    """从数据加载器中选择几个样本并返回模型的预测结果"""
    model.eval()
    all_samples = []
    all_predictions = []
    all_labels = []
    
    # 需要收集的单个样本数量
    samples_collected = 0
    
    with torch.no_grad():
        for batch in data_loader:
            text, label = batch
            batch_size = text.size(0)
            
            # 确保我们不超过要收集的样本数
            samples_to_take = min(batch_size, num_samples - samples_collected)
            if samples_to_take <= 0:
                break
            
            # 只取batch中的前samples_to_take个样本
            text_subset = text[:samples_to_take].to(device)
            label_subset = label[:samples_to_take]
            
            output = model(text_subset)
            
            # 如果是二分类问题
            if output.shape[1] == 1 if len(output.shape) > 1 else False:
                pred = torch.round(torch.sigmoid(output)).cpu()
            else:
                pred = output.argmax(dim=1, keepdim=True).cpu() if len(output.shape) > 1 else output.unsqueeze(1).cpu()
            
            # 将样本逐个添加到结果列表
            for i in range(samples_to_take):
                all_samples.append(text_subset[i].cpu())
                all_predictions.append(pred[i])
                all_labels.append(label_subset[i])
            
            samples_collected += samples_to_take
            if samples_collected >= num_samples:
                break
    
    return all_samples, all_predictions, all_labels

# 如果测试加载器不为空，获取一些样本预测结果
if list(test_loader) and len(test_loader.dataset) > 0:
    # 确保RNN和LSTM模型已经加载和训练好
    if 'rnn_model' in locals() and 'lstm_model' in locals():
        num_samples = min(5, len(test_loader))  # 取最多5个样本或全部样本
        
        print("获取RNN模型预测...")
        rnn_samples, rnn_preds, rnn_labels = predict_samples(rnn_model, test_loader, num_samples)
        
        print("获取LSTM模型预测...")
        lstm_samples, lstm_preds, lstm_labels = predict_samples(lstm_model, test_loader, num_samples)
        
        # 打印每个样本的预测结果对比
        print("\n预测结果对比:")
        print("=" * 50)
        for i in range(len(rnn_samples)):
            print(f"样本 {i+1}:")
            true_label = rnn_labels[i].squeeze().item() if rnn_labels[i].numel() == 1 else rnn_labels[i][0].item()
            rnn_pred = rnn_preds[i].squeeze().item() if rnn_preds[i].numel() == 1 else rnn_preds[i][0].item()
            lstm_pred = lstm_preds[i].squeeze().item() if lstm_preds[i].numel() == 1 else lstm_preds[i][0].item()
            
            print(f"真实标签: {true_label}")
            print(f"RNN 预测: {rnn_pred}, {'正确' if rnn_pred == true_label else '错误'}")
            print(f"LSTM 预测: {lstm_pred}, {'正确' if lstm_pred == true_label else '错误'}")
            print("-" * 50)
            
        # 计算模型间的预测一致性
        agreement_count = sum(1 for rp, lp in zip(rnn_preds, lstm_preds) if rp.item() == lp.item())
        agreement_rate = agreement_count / len(rnn_preds)
        print(f"RNN 和 LSTM 预测一致率: {agreement_rate:.2%}")
    else:
        print("RNN或LSTM模型未定义，无法进行预测分析。")
else:
    print("测试数据加载器为空，无法获取样本进行预测分析。")

# 结论与改进方向已在原Markdown单元格中讨论

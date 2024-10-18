import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
from matplotlib import font_manager

my_font = font_manager.FontProperties(fname="/Library/Fonts/Songti.ttc")  # 设置matplotlib绘图时的字体
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

# 导入数据
file_name1 = r'train.csv'
file_name2 = r'test.csv'
train_data = pd.read_csv(file_name1, sep=r'\t', header=None, engine="python", encoding="gbk")
test_data = pd.read_csv(file_name2, sep=r'\t', header=None, engine="python", encoding="gbk")

# 对训练数据集中的第二列进行计数，并将结果以柱状图的形式展示出来
df_cnt = pd.DataFrame(train_data[1].value_counts()).reset_index()
df_cnt.columns = ['cat', 'cnt']
df_cnt.plot(x='cat', y='cnt', kind='bar', legend=False, figsize=(8, 5))
plt.title("类目分布")
plt.ylabel('数量', fontsize=18)
plt.xlabel('类目', fontsize=18)
plt.show()

# 重命名训练数据集中的第二列为'cat'，并为其创建一个新的列'cat_id'，并将其转换为分类变量。
train_data.columns = ['text', 'cat']
train_data['cat_id'] = train_data['cat'].factorize()[0]
# 为测试数据集创建'cat'和'cat_id'列
test_data.columns = ['text']
test_data['cat'] = -1
test_data['cat_id'] = -1
# 将训练数据集和测试数据集合并到一起
data = train_data
print(train_data.shape, test_data.shape, data.shape)
# 创建一个字典，将类别映射到类别ID，以及将类别ID映射回类别
cat_id_df = data[['cat', 'cat_id']].drop_duplicates().sort_values('cat_id').reset_index(drop=True)
cat_to_id = dict(cat_id_df.values)
id_to_cat = dict(cat_id_df[['cat_id', 'cat']].values)


# 定义删除除字母,数字，汉字以外的所有符号的函数
def remove_punctuation(line):
    import re
    line = str(line)
    if line.strip() == '':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('', line)
    return line


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


import jieba
import csv

# 加载停用词
stopwords = ' '.join(
    pd.read_csv(r'停用词.txt', header=None, skiprows=1, quoting=csv.QUOTE_NONE)[0])

# 删除除字母,数字，汉字以外的所有符号
data['clean_text'] = data['text'].apply(remove_punctuation)
data.sample(10)

# 分词，并过滤停用词
data['cut_text'] = data['clean_text'].apply(lambda x: " ".join([w for w in list(jieba.cut(x)) if w not in stopwords]))

# 自定义词汇表
tokenizer = {}
word_index = {}
index = 1
for sentence in data['cut_text']:
    for word in sentence.split():
        if word not in tokenizer:
            tokenizer[word] = index
            word_index[index] = word
            index += 1
tokenizer['<PAD>'] = 0
word_index[0] = '<PAD>'
vocab_size = len(tokenizer) + 1  # 包括填充字符


# 将文本转换为序列
def text_to_sequence(texts, tokenizer, max_length):
    sequences = []
    for text in texts:
        sequence = [tokenizer[word] for word in text.split() if word in tokenizer]
        sequence = sequence[:max_length]  # 截断
        sequence += [0] * (max_length - len(sequence))  # 填充
        sequences.append(sequence)
    return sequences


MAX_SEQUENCE_LENGTH = 250
X = text_to_sequence(data['cut_text'], tokenizer, MAX_SEQUENCE_LENGTH)
Y = data['cat_id'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)


# PyTorch 数据集定义
class TextDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        return {
            'sequence': torch.tensor(sequence, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }


train_dataset = TextDataset(X_train, Y_train)
test_dataset = TextDataset(X_test, Y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 定义LSTM模型
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)

        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        return self.fc(hidden)


# 训练参数
EMBEDDING_DIM = 100
HIDDEN_DIM = 100
OUTPUT_DIM = len(cat_to_id)
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LSTMClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# 训练循环
epochs = 5
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        input_ids = batch['sequence'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        predictions = model(input_ids)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# 评估模型
model.eval()
with torch.no_grad():
    correct, total = 0, 0
    all_preds, all_labels = [], []
    for batch in test_loader:
        input_ids = batch['sequence'].to(device)
        labels = batch['labels'].to(device)

        predictions = model(input_ids)
        _, predicted = torch.max(predictions, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

# 绘制混淆矩阵和其他可视化
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

conf_mat = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=cat_id_df.cat.values, yticklabels=cat_id_df.cat.values)
plt.ylabel('实际结果', fontsize=18)
plt.xlabel('预测结果', fontsize=18)
plt.show()

print('Accuracy: %s' % accuracy_score(all_labels, all_preds))
print(classification_report(all_labels, all_preds, target_names=cat_id_df['cat'].values))


# 预测函数
def predict(text):
    txt = remove_punctuation(text)
    txt = " ".join([w for w in list(jieba.cut(txt)) if w not in stopwords])
    sequence = [tokenizer.get(word, 0) for word in txt.split()]  # 使用未知词的索引作为默认值
    sequence = sequence[:MAX_SEQUENCE_LENGTH]  # 截断
    sequence += [0] * (MAX_SEQUENCE_LENGTH - len(sequence))  # 填充

    with torch.no_grad():
        model.eval()
        tensor_sequence = torch.tensor(sequence, dtype=torch.long).unsqueeze(0).to(device)
        predictions = model(tensor_sequence)
        _, predicted_class = torch.max(predictions, 1)
        res = id_to_cat[predicted_class.item()]
        return res

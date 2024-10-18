import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup
import warnings
import seaborn as sns
from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import confusion_matrix

# 设置最频繁使用的50000个词
MAX_NB_WORDS = 50000#指定词汇表的大小
# 每条cut_review最大的长度
MAX_SEQUENCE_LENGTH = 250#指定每条文本的最大长度
# 设置Embeddingceng层的维度
EMBEDDING_DIM = 100#指定词向量的维度
import jieba
import csv
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

# 加载停用词
stopwords =' '.join(pd.read_csv(r'C:\Users\lenovo\Desktop\停用词.txt', header=None, skiprows=1, quoting=csv.QUOTE_NONE)[0])

import matplotlib.pyplot as plt   #python3.7
plt.rcParams['font.sans-serif']=['SimHei']   #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False   #用来正常显示负号
from matplotlib import font_manager
my_font = font_manager.FontProperties(fname="/Library/Fonts/Songti.ttc")  # 设置matplotlib绘图时的字体
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

# 导入数据
file_name1 = r'train.csv'
file_name2 = r'test.csv'
train_data = pd.read_csv(file_name1, sep=r'\t', header=None, engine="python", encoding="gbk")
test_data = pd.read_csv(file_name2, sep=r'\t', header=None, engine="python", encoding="gbk")

#对训练数据集中的第二列进行计数，并将结果以柱状图的形式展示出来
df_cnt = pd.DataFrame(train_data[1].value_counts()).reset_index()
df_cnt.columns=['cat','cnt']
df_cnt.plot(x='cat', y='cnt', kind='bar', legend=False,  figsize=(8, 5))
plt.title("类目分布")
plt.ylabel('数量', fontsize=18)
plt.xlabel('类目', fontsize=18)
plt.show()

# 类别ID映射相关代码保持不变...
#重命名训练数据集中的第二列为'cat'，并为其创建一个新的列'cat_id'，并将其转换为分类变量。
train_data.columns = ['text', 'cat']
train_data['cat_id'] = train_data['cat'].factorize()[0]
#为测试数据集创建'cat'和'cat_id'列
test_data.columns = ['text']
test_data['cat'] = -1
test_data['cat_id'] = -1
#将训练数据集和测试数据集合并到一起
data = train_data
print(train_data.shape, test_data.shape, data.shape)
#创建一个字典，将类别映射到类别ID，以及将类别ID映射回类别
cat_id_df = data[['cat', 'cat_id']].drop_duplicates().sort_values('cat_id').reset_index(drop=True)
cat_to_id = dict(cat_id_df.values)
id_to_cat = dict(cat_id_df[['cat_id', 'cat']].values)

epochs = 5
# 定义TextDataset类以处理BERT模型所需的输入格式
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# BERT相关部分
model_local_path = 'C:\\Users\\lenovo\\.cache\\huggingface\\hub\\models--bert-base-chinese'
vocab_file = f'{model_local_path}/vocab.txt'  # 根据实际下载的分词器文件名调整

tokenizer = BertTokenizer.from_pretrained(vocab_file, do_lower_case=True)  # 对于中文BERT，不需要转换为小写
model = BertForSequenceClassification.from_pretrained(model_local_path, num_labels=len(id_to_cat))
MAX_LEN = 128  # 设置序列最大长度

# 预处理训练数据
train_texts = data['text'].tolist()
train_labels = data['cat_id'].tolist()
train_dataset = TextDataset(train_texts, train_labels, tokenizer, MAX_LEN)

# 预处理测试数据
test_texts = test_data['text'].tolist()
test_dataset = TextDataset(test_texts, [-1] * len(test_texts), tokenizer, MAX_LEN)

# 划分训练集和验证集
train_dataset, val_dataset, train_labels, val_labels = train_test_split(
    train_dataset, train_labels, test_size=0.10, random_state=42)

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# 定义训练函数和优化器
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * epochs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 微调BERT模型
for epoch in range(epochs):
    model.train()
    # 添加训练进度条（适用于Jupyter Notebook）
    epoch_loss = 0
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', position=0, leave=True) as pbar:
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

            pbar.update(1)

        # 更新平均损失值显示
        avg_loss = epoch_loss / len(train_loader)
        pbar.set_description(f'Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}')

    # 在验证集上进行评估
    model.eval()
    val_losses = []
    with torch.no_grad():
        for val_batch in val_loader:
            val_input_ids = val_batch['input_ids'].to(device)
            val_attention_mask = val_batch['attention_mask'].to(device)
            val_labels = val_batch['labels'].to(device)

            val_outputs = model(val_input_ids, val_attention_mask, val_labels)
            val_loss = val_outputs.loss
            val_losses.append(val_loss.item())

    avg_val_loss = sum(val_losses) / len(val_loader)
    print(f'Epoch: {epoch + 1}, Validation Loss: {avg_val_loss:.4f}')

# 模型测试部分
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        predictions.extend(preds.tolist())
        true_labels.extend(batch['labels'].tolist())

# 计算准确率和混淆矩阵
y_pred = predictions
y_true = [id_to_cat[label] for label in true_labels]

conf_mat = confusion_matrix(y_true, y_pred, labels=list(id_to_cat.values()))
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=id_to_cat.values(), yticklabels=id_to_cat.values())
plt.ylabel('实际结果', fontsize=18)
plt.xlabel('预测结果', fontsize=18)
plt.show()

print('Accuracy:', accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=list(id_to_cat.keys())))

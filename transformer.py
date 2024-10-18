import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from matplotlib import font_manager

my_font = font_manager.FontProperties(fname="/Library/Fonts/Songti.ttc")
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
from tf_keras import layers, models, optimizers
warnings.filterwarnings('ignore')
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import csv
import jieba
import os

# 导入数据
file_name1 = r'train.csv'
file_name2 = r'test.csv'
train_data = pd.read_csv(file_name1, sep=r'\t', header=None, engine="python", encoding="gbk")
test_data = pd.read_csv(file_name2, sep=r'\t', header=None, engine="python", encoding="gbk")

# 重命名列并创建类别ID
train_data.columns = ['text', 'cat']
train_data['cat_id'] = train_data['cat'].factorize()[0]
test_data.columns = ['text']
test_data['cat'] = -1
test_data['cat_id'] = -1
data = train_data
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


# 加载停用词
stopwords = ' '.join(
    pd.read_csv(r'F:\\ZJD\\bs\\桌面数据集、图标\\停用词.txt', header=None, skiprows=1, quoting=csv.QUOTE_NONE)[0])

# 删除除字母,数字，汉字以外的所有符号
data['clean_text'] = data['text'].apply(remove_punctuation)

# 分词，并过滤停用词
data['cut_text'] = data['clean_text'].apply(lambda x: " ".join([w for w in list(jieba.cut(x)) if w not in stopwords]))

# 初始化tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 准备数据
X = data['cut_text'].values
Y = data['cat_id'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)


# 创建PyTorch Dataset
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# 设置最大长度
MAX_SEQUENCE_LENGTH = 250

# 创建数据集
train_dataset = TextDataset(X_train, Y_train, tokenizer, MAX_SEQUENCE_LENGTH)
test_dataset = TextDataset(X_test, Y_test, tokenizer, MAX_SEQUENCE_LENGTH)

# 创建模型
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=len(cat_to_id))

# 打印当前工作目录
print(f"Current working directory: {os.getcwd()}")

# 确保 output_dir 和 logging_dir 存在，并使用绝对路径
output_dir = os.path.abspath('./results')
logging_dir = os.path.abspath('./logs')

for dir_path in [output_dir, logging_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

print(f"Output directory: {output_dir}")
print(f"Logging directory: {logging_dir}")

# 训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch"
)

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# 开始训练
trainer.train()

# 评估模型
trainer.evaluate()


# 预测函数
def predict(text):
    txt = remove_punctuation(text)
    txt = " ".join([w for w in list(jieba.cut(txt)) if w not in stopwords])
    encoding = tokenizer.encode_plus(
        txt,
        add_special_tokens=True,
        max_length=MAX_SEQUENCE_LENGTH,
        padding='max_length',
        truncation=True,
        return_token_type_ids=False,
        return_attention_mask=True,
        return_tensors='pt',
    ).to(device)

    with torch.no_grad():
        model.eval()
        predictions = model(**encoding)
        _, predicted_class = torch.max(predictions.logits, 1)
        res = id_to_cat[predicted_class.item()]
        return res

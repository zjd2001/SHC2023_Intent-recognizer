import matplotlib.pyplot as plt   #python3.7
plt.rcParams['font.sans-serif']=['SimHei']   #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False   #用来正常显示负号
from matplotlib import font_manager
my_font = font_manager.FontProperties(fname="/Library/Fonts/Songti.ttc")  # 设置matplotlib绘图时的字体
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

#导入数据
file_name1=r'train1.csv'
file_name2=r'test.csv'
train_data = pd.read_csv(file_name1,sep=r'\t',header=None,engine="python",encoding="gbk")
test_data = pd.read_csv(file_name2,sep=r'\t',header=None,engine="python",encoding="gbk")

#对训练数据集中的第二列进行计数，并将结果以柱状图的形式展示出来
df_cnt = pd.DataFrame(train_data[1].value_counts()).reset_index()
df_cnt.columns=['cat','cnt']
df_cnt.plot(x='cat', y='cnt', kind='bar', legend=False,  figsize=(8, 5))
plt.title("类目分布")
plt.ylabel('数量', fontsize=18)
plt.xlabel('类目', fontsize=18)
plt.show()

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
stopwords =' '.join(pd.read_csv(r'停用词.txt', header=None, skiprows=1, quoting=csv.QUOTE_NONE)[0])

# 删除除字母,数字，汉字以外的所有符号
data['clean_text'] = data['text'].apply(remove_punctuation)
data.sample(10)

# 分词，并过滤停用词
data['cut_text'] = data['clean_text'].apply(lambda x: " ".join([w for w in list(jieba.cut(x)) if w not in stopwords]))

# LSTM 模型构建
## 将cut_text数据进行向量化处理
## 设置最频繁使用的50000个词 设置每条 cut_text最大的词语数为250个(超过的将会被截去,不足的将会被补0)
# 设置最频繁使用的50000个词
MAX_NB_WORDS = 50000#指定词汇表的大小
# 每条cut_review最大的长度
MAX_SEQUENCE_LENGTH = 250#指定每条文本的最大长度
# 设置Embeddingceng层的维度
EMBEDDING_DIM = 100#指定词向量的维度

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(data['cut_text'].values)
word_index = tokenizer.word_index
print('共有 %s 个不相同的词语.' % len(word_index))

X = tokenizer.texts_to_sequences(data['cut_text'].values)
# 填充X,让X的各个列的长度统一
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)#将整数序列填充到相同的长度
# 多类标签的onehot展开
Y = pd.get_dummies(data['cat_id']).values#将多类标签展开为one-hot编码
print(X.shape)
print(Y.shape)

# 拆分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from keras.callbacks import EarlyStopping

model = Sequential()#初始化一个Sequential模型
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))#添加一个嵌入层
model.add(SpatialDropout1D(0.2)) 

model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(9, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
epochs =5
batch_size = 64
model_lstm = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1) #callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)]

#######################模型训练后测试集准确率展示###############################
import seaborn as sns
from sklearn.metrics import confusion_matrix
#导入了seaborn和sklearn.metrics库，分别用于绘制热力图和计算准确率、混淆矩阵

y_pred = model.predict(X_test)
y_pred = y_pred.argmax(axis=1)
Y_test = Y_test.argmax(axis=1)
#使用模型对测试数据集X_test进行预测，将预测结果存储在y_pred变量中
#通过argmax方法找到y_pred和Y_test每行最大值所对应的索引，将其转换为分类标签形式

conf_mat = confusion_matrix(Y_test, y_pred)
#使用sklearn.metrics库中的confusion_matrix方法计算混淆矩阵，传入真实标签Y_test和预测结果y_pred作为参数，结果存储在conf_mat变量中
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=cat_id_df.cat.values, yticklabels=cat_id_df.cat.values)
#创建一个大小为(10, 8)的子图，并使用seaborn库的heatmap方法绘制热力图。
#其中，传入conf_mat作为数据，设置annot=True表示在热力图上显示数值，
#fmt='d'表示以整数形式显示，xticklabels和yticklabels用于设置刻度标签，根据cat_id_df.cat.values的取值。
plt.ylabel('实际结果', fontsize=18)#设置纵轴标签'实际结果'和横轴标签'预测结果'的字体大小为18
plt.xlabel('预测结果', fontsize=18)
plt.show()

from  sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

print('accuracy %s' % accuracy_score(y_pred, Y_test))
#通过将预测结果y_pred和真实标签Y_test作为参数传递给accuracy_score函数，可以计算出准确率
print(classification_report(Y_test, y_pred,target_names=cat_id_df['cat'].values))
#classification_report函数用于生成分类模型的评估报告。
#该函数接受真实标签Y_test和预测结果y_pred作为参数，并使用target_names参数传递类别名称。
#使用了cat_id_df['cat'].values作为类别名称。
print(model_lstm.history['loss'])
print(model_lstm.history['val_loss'])
print(model_lstm.history['accuracy'])
print(model_lstm.history['val_accuracy'])
plt.title('Loss')
plt.plot(model_lstm.history['loss'], label='train', marker='^',color='k',linewidth=1.8)
plt.plot(model_lstm.history['val_loss'], label='test', marker='o',color='r',linewidth=1.8)
plt.legend()
plt.show()

plt.title('Accuracy')

plt.plot(model_lstm.history['accuracy'], label='train',marker='^',color='k',linewidth=1.8)
plt.plot(model_lstm.history['val_accuracy'], label='test', marker='o',color='r',linewidth=1.8)
plt.legend()
plt.show()

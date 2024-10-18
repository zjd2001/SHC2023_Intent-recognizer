import pandas as pd

file_name1=r'train.csv'
file_name2=r'test.csv'
train_data = pd.read_csv(file_name1,sep=r'\t',header=None,engine="python",encoding="gbk")
test_data = pd.read_csv(file_name2,sep=r'\t',header=None,engine="python",encoding="gbk")

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import jieba
import csv

stoplist=' '.join(pd.read_csv(r'C:\Users\lenovo\Desktop\停用词.txt', header=None, skiprows=1, quoting=csv.QUOTE_NONE)[0])
#设定分词及清理停用词函数
def m_cut(intxt):
    return [w for w in jieba.lcut(intxt) if w not in stoplist and len(w) > 1]

def tfidf_cnt(category):
    txt_list = []
    corpus =list(train_data[train_data[1] == category][0])   # len=1304
    for w in corpus:
        txt_list+=[" ".join(m_cut(w))]      # 用空格分隔开,才能传入fit_transform中
    # txt_list
    vectorizer = CountVectorizer()         # 创建计算词频的实例
    X = vectorizer.fit_transform(txt_list) # 将文本中的词语转换为词频稀疏矩阵
    transformer = TfidfTransformer()  # 初始化TF-IDF
    tfidf = transformer.fit_transform(X)  #基于词频稀疏矩阵X计算TF-IDF值
    word=vectorizer.get_feature_names_out()#获取词袋模型中的所有词语
    weight=tfidf.toarray()               #将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    data_dict = {}
    for i in range(len(weight)):     #文本的tf-idf词语权重添加到字典data_dict中
        for j in range(len(word)):
            data_dict[word[j]] = weight[i,j]
#     sorted(data_dict.items(),key=lambda x:x[1],reverse=True)[:7]  #按照tfidf值倒叙输出前7个
    return pd.DataFrame(sorted(data_dict.items(),key=lambda x:x[1],reverse=True)[:7])

df_cnt = pd.DataFrame()
for i in list(set(train_data[1])):
    df = tfidf_cnt(i)
    df['category'] = i
    df_cnt = pd.concat([df_cnt,df],axis=0)

# 载入停用词
stopwords = set()

with open(r'停用词.txt','w+', encoding='utf8') as infile:

    for line in infile:
        line = line.rstrip('\n')
        if line:
            stopwords.add(line.lower())

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(tokenizer=jieba.lcut, stop_words=stopwords, min_df=50, max_df=0.3)
x = tfidf.fit_transform(train_data[0])

# 训练分类器   编码目标变量,sklearn只接受数值
from sklearn.preprocessing import LabelEncoder  # LabelEncoder：将类别数据数字化
from sklearn.model_selection import train_test_split  # 分割数据集

y_encoder = LabelEncoder()
y = y_encoder.fit_transform(train_data[1])  # 将类别转换成0,1,2,3,4,5,6,7,8,9...

# 根据y分层抽样，测试数据占20%  stratify的作用：保持测试集与整个数据集里result的数据分类比例一致。  pd.concat([pd.DataFrame(train_y).value_counts(),pd.DataFrame(test_y).value_counts()],axis=1)
train_idx, test_idx = train_test_split(range(len(y)), test_size=0.2, stratify=y)
train_x = x[train_idx, :]
train_y = y[train_idx]
test_x = x[test_idx, :]
test_y = y[test_idx]

# 训练逻辑回归模型 我们是12分类  属于多分类

from sklearn.linear_model import LogisticRegression  # 引入逻辑回归

clf_lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')  # solver='lbfgs'：求解方式
clf_lr.fit(train_x, train_y)

#  常用参数说明
#  penalty: 正则项类型，l1还是l2
#  C: 正则项惩罚系数的倒数，越大则惩罚越小
#  fit_intercept: 是否拟合常数项
#  max_iter: 最大迭代次数
#  multi_class: 以何种方式训练多分类模型
#     ovr = 对每个标签训练二分类模型
#     multinomial ovo = 直接训练多分类模型，仅当solver={newton-cg, sag, lbfgs}时支持
#     solver: 用哪种方法求解，可选有{liblinear, newton-cg, sag, lbfgs}
#      小数据liblinear比较好，大数据量sag更快
#      多分类问题，liblinear只支持ovr模式，其他支持ovr和multinomial
#      liblinear支持l1正则，其他只支持l2正则


# 模型效果评估
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

y_pred_lr = clf_lr.predict(test_x)
pd.DataFrame(confusion_matrix(test_y, y_pred_lr), columns=y_encoder.classes_, index=y_encoder.classes_)

import numpy as np
# 计算各项评价指标
def eval_model(y_true, y_pred, labels):
    # 计算每个分类的Precision, Recall, f1, support
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred)
    # 计算总体的平均Precision, Recall, f1, support
    tot_p = np.average(p, weights=s)
    tot_r = np.average(r, weights=s)
    tot_f1 = np.average(f1, weights=s)
    tot_s = np.sum(s)
    res1 = pd.DataFrame({
        u'Label': labels,
        u'Precision': p,
        u'Recall': r,
        u'F1': f1,
        u'Support': s
    })
    res2 = pd.DataFrame({
        u'Label': [u'总体'],
        u'Precision': [tot_p],
        u'Recall': [tot_r],
        u'F1': [tot_f1],
        u'Support': [tot_s]
    })
    res2.index = [999]
    res = pd.concat([res1, res2])
    return res[[u'Label', u'Precision', u'Recall', u'F1', u'Support']]



from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

clf_knn = KNeighborsClassifier(n_neighbors=12)
clf_knn .fit(train_x, train_y)
y_pred_knn = clf_knn .predict(test_x)

clf_svm = svm.SVC(C=1, kernel='rbf', gamma=1, decision_function_shape='ovo')
clf_svm.fit(train_x, train_y)
y_pred_svm = clf_svm.predict(test_x)

eval_model(test_y, y_pred_svm, y_encoder.classes_)

import os

output_dir = u'output'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


from pyexpat import model

import dill
import pickle

model_file = os.path.join(output_dir, u'model.pkl')
with open(model_file, 'wb') as outfile:
    dill.dump({
        'y_encoder': y_encoder,
        'tfidf': tfidf,
        'lr': model
    }, outfile)


import pickle


model_file = os.path.join(output_dir, u'model.pkl')  # 加载模型
with open(model_file, 'rb') as infile:
    model = pickle.load(infile)

# 转化为词袋表示
new_x = model['tfidf'].transform(test_data[0][:50])

# 预测类别
new_y_pred = model['lr'].predict(new_x)

# 解释类别
pd.DataFrame({u'预测类别': model['y_encoder'].inverse_transform(new_y_pred), u'文本': test_data[0][:50]})

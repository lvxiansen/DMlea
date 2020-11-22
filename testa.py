import os
import jieba
# 将特征与类别组合并打乱，再随机抽取训练集和测试集
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import sys
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
def get_stopword():
    #创建一个无序不重复元素集
    stopword_set = set()
    with open("./stop_words.txt", 'r',encoding='gbk') as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip("\n"))
            #print(stopword_set)
    return stopword_set
# 读取所有文件并组成矩阵，特征和类别单独存放
fold_path = r"D:\Naive-Bayes-Text-Classifier\Database\SogouC\Sample"
folder_list = os.listdir(fold_path)  # 读取文件夹列表
#print(folder_list)
artcilt_list = []
class_list = []
stopwords = get_stopword()
for fold in folder_list:  # 读取子文件夹列表
    new_fold_path = os.path.join(fold_path,fold) # 将路径拼接
    files = os.listdir(new_fold_path) # 再读取子文件夹
    for file in files: # 读取文件
        with open(os.path.join(new_fold_path,file),'r',encoding='utf-8') as fp:
            article = fp.read()
            article1 =' '.join(jieba.cut(article,cut_all=False)) # 精确模式分词
            artcilt_list.append(article1) # 组成列表
            class_list.append(fold)
test_size = 0.2


np.set_printoptions(threshold=sys.maxsize)  # 设置显示矩阵所有行列
# tfidf_transformer = TfidfVectorizer()
# tf_train_data = tfidf_transformer.fit_transform(["这个!不是!这么!做的",'话说!学校'])

#也可以用sklearn库中的train_test_split打乱并随机抽取特定比例的训练集和测试集
from sklearn.model_selection import train_test_split
train_data,test_data,train_class,test_class = train_test_split(artcilt_list,class_list,test_size=test_size,shuffle=True)

tfidf = TfidfVectorizer()
tf_train_data = tfidf.fit_transform(train_data)

clf = SVC(kernel = 'linear').fit(tf_train_data, train_class)
tf_test_data = tfidf.transform(test_data)
y_pred = clf.predict(tf_test_data)
print("---")
print(y_pred)
print("---")
print('训练集分数:', clf.score(tf_train_data, train_class))
print('测试集分数:', metrics.accuracy_score(y_pred, test_class))
print('混淆矩阵:')
print(confusion_matrix(y_pred, test_class))
print('分类报告:', classification_report(y_pred, test_class))


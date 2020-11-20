import os
import jieba
# 将特征与类别组合并打乱，再随机抽取训练集和测试集
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import sys
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


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
            """
            article = fp.read()
            article1 = jieba.lcut(article, cut_all=False)
            article2 = ""
            for i in range(len(article1)):
                if article1[i] not in stopwords:
                    article2 += article1[i]+" "
            artcilt_list.append(article2) # 组成列表
            class_list.append(fold)
            """
            article = fp.read()
            article1 =' '.join(jieba.cut(article,cut_all=False)) # 精确模式分词
            artcilt_list.append(article1) # 组成列表
            class_list.append(fold)
#print(artcilt_list)
#print(class_list)


# # data_list = list(zip(artcilt_list, class_list))
# # random.shuffle(data_list)  # 打乱顺序
test_size = 0.2
# # index = int(len(data_list) * test_size) + 1
# # train_data_list = data_list[:index]
# # #print(train_data_list)
# # test_data_list = data_list[index:]
# # print(test_data_list)
# # train_data, train_class = zip(*train_data_list)
# # test_data, test_class = zip(*test_data_list)
#

from sklearn.feature_extraction.text import TfidfVectorizer
np.set_printoptions(threshold=sys.maxsize)  # 设置显示矩阵所有行列
tfidf_transformer = TfidfVectorizer()
tf_train_data = tfidf_transformer.fit_transform(["这个!不是!这么!做的",'话说!学校'])
# print(tfidf_transformer)  # 输出参数
# print(tf_train_data)# 稀疏矩阵表示法
# print(tf_train_data.toarray()[1])  # 一般矩阵表示法
# print(tf_train_data.todense()[1])  # 一般矩阵表示法
# print("-----")
# print(tfidf_transformer.get_feature_names())  # 获得特征名
# #print(tf_train_data.idf_[0])  # 输出IDF值
# print(tfidf_transformer.vocabulary_) # 词语与列的对应关系

#也可以用sklearn库中的train_test_split打乱并随机抽取特定比例的训练集和测试集
from sklearn.model_selection import train_test_split
train_data,test_data,train_class,test_class = train_test_split(artcilt_list,class_list,test_size=test_size,shuffle=True)
#
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# vector = CountVectorizer()
# train_data = vector.fit_transform(["zhe rshi ge hao shu ji",'zhe bu shi'])
# np.set_printoptions(threshold=sys.maxsize)  # 设置显示矩阵所有行列
# print('特征个数:', len(vector.get_feature_names()))
# print('特征词:', vector.vocabulary_)
# print('词频:', train_data.toarray())
# transfomer = TfidfTransformer()
# tf_train_data = transfomer.fit_transform(train_data)
# print('词频矩阵', tf_train_data)
# print('词频矩阵', tf_train_data.toarray())
# # 建立tf-idf词频权重矩阵
#
#
tfidf = TfidfVectorizer()
tf_train_data = tfidf.fit_transform(train_data)
# for i in range(len(train_data)):
#     print(train_data[i])
# # print("!!!!!!")
# np.set_printoptions(threshold=sys.maxsize)
# #print(tfidf.vocabulary_)  # 输出词典及位置
# #print(tfidf.idf_)  # 输出逆向文件频率
#
# 用贝叶斯多项式模型分类并输出分类结果


clf = MultinomialNB(fit_prior=True)
# print("!!!")
# print(tf_train_data[0])
# print("!!!!!")
# print(tf_train_data)
# print(train_class)
# print(len(train_class))
print("------")
print(tf_train_data.shape[0])
print("---")
clf.fit(tf_train_data, train_class)
tf_test_data = tfidf.transform(test_data)
y_pred = clf.predict(tf_test_data)
# print("---")
# print(y_pred)
# print("---")
# print('训练集分数:', clf.score(tf_train_data, train_class))
# print('测试集分数:', metrics.accuracy_score(y_pred, test_class))
# print('混淆矩阵:')
print(confusion_matrix(y_pred, test_class))
print('分类报告:', classification_report(y_pred, test_class))


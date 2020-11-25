import os
import jieba
import time
import re
import joblib
import pickle
# 将特征与类别组合并打乱，再随机抽取训练集和测试集
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import sys
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
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
# fold_path = r"D:\Naive-Bayes-Text-Classifier\Database\SogouC\Sample"
# folder_list = os.listdir(fold_path)  # 读取文件夹列表
#print(folder_list)
# artcilt_list = []
# class_list = []
#stopwords = get_stopword()
# for fold in folder_list:  # 读取子文件夹列表
#     new_fold_path = os.path.join(fold_path,fold) # 将路径拼接
#     files = os.listdir(new_fold_path) # 再读取子文件夹
#     for file in files: # 读取文件
#         with open(os.path.join(new_fold_path,file),'r',encoding='utf-8') as fp:
#             article = fp.read()
#             article1 =' '.join(jieba.cut(article,cut_all=False)) # 精确模式分词
#             artcilt_list.append(article1) # 组成列表
#             class_list.append(fold)
test_size = 0.5

def get_classify_count(path):
    labels = []
    texts = []
    classes = ['it', 'auto', 'stock', 'yule', 'sports', 'business', 'health', 'learning', 'women', 'house']
    start_time = time.time()

    filea = open(path, encoding='UTF-8')
    articals = filea.readlines()
    filea.close()

    fileb = open("./corpus_old.txt", encoding='UTF-8')
    article_listb = fileb.readlines()
    fileb.close()

    articals.extend(article_listb)

    stopwords = get_stopword()
    counta = 0
    for i in range(len(articals)//6):
        #对新闻标签和内容进行预处理
        url_whole = articals[i*6+1]
        content_whole = articals[i*6+4]
        url_whole = re.sub('<url>|</url>|http:\\/\\/','',url_whole)
        content = re.sub('<content>|</content>','',content_whole)
        url_list = url_whole.split('.')
        index = url_list.index('sohu')
        counta += 1
        if(len(content)>30 and len(content)<3000):
            if url_list[index-1] in classes:
                result_content = ""
                labels.append(url_list[index-1])
                words = jieba.cut(content, cut_all=False)
                for word in words:
                    if word not in stopwords:
                        result_content += word + " "
                #print(result_content )
                texts.append(result_content)
    cnt_class = list(np.zeros(10))
    for label in labels:
        for i in range(len(classes)):
            if (classes[i] == label):
                cnt_class[i] += 1
    print("总共{}篇文章".format(len(labels)))
    for i in range(len(classes)):
        print("类别:",classes[i],"个数为：",cnt_class[i])
    end_time = time.time()
    print("所用时间：{}".format(end_time-start_time))
    print("总共",counta)
    print(len(labels))
    print(len(texts))
    # with open("svmspllist.pkl", 'wb') as f:
    #     pickle.dump(labels, f)
    #     pickle.dump(texts, f)
    return texts,labels

#np.set_printoptions(threshold=sys.maxsize)  # 设置显示矩阵所有行列
#artcilt_list,class_list = get_classify_count("./cor.txt")
#也可以用sklearn库中的train_test_split打乱并随机抽取特定比例的训练集和测试集
# from sklearn.model_selection import train_test_split
#train_data,test_data,train_class,test_class = train_test_split(artcilt_list,class_list,test_size=test_size,shuffle=True)
# with open("svmtraintest.pkl", 'wb') as svmf:
#     pickle.dump(train_data, svmf)
#     pickle.dump(test_data, svmf)
#     pickle.dump(train_class, svmf)
#     pickle.dump(test_class, svmf)
#
#
def geta():
    with open("./testtrainfwords.pkl", "rb") as faaf:
        test = pickle.load(faaf)
        train = pickle.load(faaf)
    trainlabellist = []
    traintextlist = []
    for line in train:
        lable, text = line.split(' ', 1)
        trainlabellist.append(lable)
        traintextlist.append(text)
    testlabellist = []
    testtextlist = []
    for line in test:
        lable, text = line.split(' ', 1)
        testlabellist.append(lable)
        testtextlist.append(text)
    with open("svmtraintestb.pkl", 'wb') as svmf:
        pickle.dump(traintextlist, svmf)
        pickle.dump(testtextlist, svmf)
        pickle.dump(trainlabellist, svmf)
        pickle.dump(testlabellist, svmf)
geta()
with open("svmtraintestb.pkl", 'rb') as rsvmf:
    train_data = pickle.load(rsvmf)
    test_data = pickle.load(rsvmf)
    train_class = pickle.load(rsvmf)
    test_class = pickle.load(rsvmf)
tfidf = TfidfVectorizer()
voca = tfidf.vocabulary_
with open("svmvoc.pkl", 'wb') as vocw:
    pickle.dump(voca, vocw)
tf_train_data = tfidf.fit_transform(train_data)
tf_test_data = tfidf.transform(test_data)
#
with open("svmtestdata.pkl", 'wb') as svmtest:
    pickle.dump(tf_train_data, svmtest)
    pickle.dump(tf_test_data, svmtest)

with open("svmtestdata.pkl", 'rb') as rsvmtest:
    tf_train_data = pickle.load(rsvmtest)
    tf_test_data = pickle.load(rsvmtest)
print(time.time())
clf = LinearSVC()
clf.fit(tf_train_data, train_class)
print(time.time())
joblib.dump(clf, './svmmodel.pkl')

timea = time.time()
print(timea)
y_pred = clf.predict(tf_test_data)
timeb = time.time()
print(timeb)


with open("ypred.pkl", 'wb') as svmypre:
    pickle.dump(y_pred, svmypre)
print('训练集分数:', clf.score(tf_train_data, train_class))
#print('测试集分数:', metrics.accuracy_score(y_pred, test_class))
print('测试集分数:', metrics.accuracy_score(test_class,y_pred))
print('混淆矩阵:')
print(confusion_matrix(test_class,y_pred))
print('分类报告:', classification_report(test_class,y_pred))


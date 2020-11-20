import os
import re
import operator
import time
import multiprocessing
import jieba
import sys
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from collections import defaultdict
import numpy as np
def fenci(path):
    start_time = time.time()
    file = open(path, encoding='UTF-8')
    articals = file.readlines()
    file.close()
    stopwords = get_stopword()
    fenci_after = list()
    adic = defaultdict(int)
    anum = 0
    for i in range(len(articals)//6):
        #对新闻标签和内容进行预处理
        url_whole = articals[i*6+1]
        content_whole = articals[i*6+4]
        url_whole = re.sub('<url>|</url>|http:\\/\\/','',url_whole)
        content = re.sub('<content>|</content>','',content_whole)
        url_list = url_whole.split('.')
        index = url_list.index('sohu')
        # if i % 100 == 0:
        #     print("已对{}万篇文章分词".format(i / 100))
        if(len(content)>30 and len(content)<3000):
            temp_list = ""
            words = jieba.cut(content, cut_all=False)
            for word in words:
                    if word not in stopwords:
                        if(word not in adic):
                            anum += 1
                        temp_list += word + " "
            fenci_after.append(temp_list)
    print(anum)
    # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer()
    # 该类会统计每个词语的tf-idf权值
    tf_idf_transformer = TfidfTransformer()
    # 将文本转为词频矩阵并计算tf-idf
    tf_idf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(fenci_after))
    # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    x_train_weight = tf_idf.toarray()
    print('输出x_train文本向量：')
    print(x_train_weight)
    np.set_printoptions(threshold=np.inf)  # 设置显示矩阵所有行列
    print("获得特征名称",vectorizer.get_feature_names())
    print("获得词典及对应的位置下标",vectorizer.vocabulary_)
    print('特征个数:', len(vectorizer.get_feature_names()))
    print('特征词:', vectorizer.vocabulary_)
    print('词频:', vectorizer.fit_transform(fenci_after))
    sorted_classfiy = sorted(vectorizer.vocabulary_.items(), key=operator.itemgetter(1), reverse=True) #排序
    print("获得词典及对应的位置下标", vectorizer.vocabulary_)

    """
    tfidf_vec = TfidfVectorizer()
    tfidf_matrix = tfidf_vec.fit_transform(fenci_after)
    # 得到语料库所有不重复的词
    print(tfidf_vec.get_feature_names())
    # 得到每个单词对应的id值
    print(tfidf_vec.vocabulary_)
    # 得到每个句子所对应的向量，向量里数字的顺序是按照词语的id顺序来的
    print(tfidf_matrix.toarray())
    """
    end_time = time.time()
    print("所用时间：{}".format(end_time-start_time))


def get_stopword():
    #创建一个无序不重复元素集
    stopword_set = set()
    with open("./stop_words.txt", 'r',encoding='gbk') as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip("\n"))
            #print(stopword_set)
    return stopword_set

if __name__ == '__main__':
    #join阻塞
    fenci(".\corpus.txt")
    #get_stopword()
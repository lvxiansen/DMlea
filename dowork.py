import os
import re
import operator
import time
import multiprocessing
import jieba
import sys
import pickle
#import pandas as pd
sys.setrecursionlimit(1000000)
def get_classify_count(path):
    print("bbbbbbbbb");
    labels = []
    texts = []
    classes = ['it', 'auto', 'stock', 'yule', 'sports', 'business', 'health', 'learning', 'women', 'travel', 'house']
    start_time = time.time()
    #classify_count = {} #各类数量
    artical_wholecount = 0 #文章总数
    over_thresold = 0 #超过阈值文章总数
    less_thresold = 0 #小于阈值文章总数
    file = open(path, encoding='UTF-8')
    articals = file.readlines()
    file.close()
    stopwords = get_stopword()
    for i in range(len(articals)//6):
        #对新闻标签和内容进行预处理
        url_whole = articals[i*6+1]
        content_whole = articals[i*6+4]
        url_whole = re.sub('<url>|</url>|http:\\/\\/','',url_whole)
        content = re.sub('<content>|</content>','',content_whole)
        url_list = url_whole.split('.')
        index = url_list.index('sohu')
        #筛选阈值下的新闻并统计各个标签个数
        #class_count = []
        if i % 100 == 0:
            print("已对{}万篇文章分词".format(i / 100))
        #for i in range(11):
            #class_count.append(0)
        if(len(content)>30 and len(content)<3000):
            if url_list[index-1] in classes:
                line = content
                result_content = ""
                labels.append(url_list[index-1])
                words = jieba.cut(line, cut_all=False)
                for word in words:
                    if word not in stopwords:
                        #print(word)
                        result_content += word + " "
                        #print(result_content)
                texts.append(result_content)

                #print(texts)
                #classify_count[url_list[index-1]] = 1
            #else:
            #类别: it 个数为： 164884
                #classify_count[url_list[index-1]] += 1
            #artical_wholecount += 1
        # elif (len(content)<=30):
        #     less_thresold += 1
        # elif (len(content)>=3000):
        #     over_thresold += 1
    #sorted_classify_count = sorted(classify_count.items(),key=operator.itemgetter(1),reverse=True)
    #print(sorted_classify_count)
    #print(artical_wholecount)
    #print("总共{}篇大小在（30，,3000）内的文章".format(artical_wholecount))
    #print("小于30的文章数：{}".format(less_thresold))
    #print("大于3000的文章数：{}".format(over_thresold))
    # print("各类文章统计如下：")
    # for c in sorted_classify_count:
    #     print(c[0],c[1])
    #创建n个大小的数组

    # for label in labels:
    #     for i in range(len(classes)):
    #         if (classes[i]==label):
    #             class_count[i] += 1
    with open("./split_data.pkl", "rb") as f:
        labels = pickle.load(f)
        texts = pickle.load(f)
    print("总共{}篇文章".format(len(labels)))
    for i in range(len(classes)):
        print("类别:",classes[i],"个数为：",class_count[i])
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
    get_classify_count(".\cor.txt")
    #get_stopword()
import os
import re
import operator
import time
import pandas as pd
import multiprocessing
from collections import defaultdict
import datetime
import jieba
import sys
import pickle
import numpy as np
import random
import math
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score


def get_classify_count(path):
    labels = []
    texts = []
    classes = ['it', 'auto', 'stock', 'yule', 'sports', 'business', 'health', 'learning', 'women', 'house']
    start_time = time.time()

    filea = open(path, encoding='UTF-8')
    articals = filea.readlines()
    filea.close()

    fileb = open("corpus_old.txt", encoding='UTF-8')
    article_listb = fileb.readlines()
    fileb.close()

    fileb = open("corpus_old.txt", encoding='UTF-8')
    article_listb = fileb.readlines()
    fileb.close()

    filec = open("tempall.txt", encoding='UTF-8')
    article_listc = filec.readlines()
    filec.close()

    articals.extend(article_listb)
    articals.extend(article_listc)
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
    with open("split_data.pkl", 'wb') as f:
        pickle.dump(labels, f)
        pickle.dump(texts, f)
def get_stopword():
    #创建一个无序不重复元素集
    stopword_set = set()
    with open("./stopwords.txt", 'r',encoding='utf-8') as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip("\n"))
            #print(stopword_set)
    return stopword_set
    # 获取对应类别的索引下标值
def lable2id(label):
    classes = ['it', 'auto', 'stock', 'yule', 'sports', 'business', 'health', 'learning', 'women', 'house']
    for i in range(len(classes)):
        if label == classes[i]:
            return i
    raise Exception('Error label %s' % (label))
def doc_dict():
    classes = ['it', 'auto', 'stock', 'yule', 'sports', 'business', 'health', 'learning', 'women', 'house']
    return [0] * len(classes)
def tfidf():
    """
    定义一个二维数组
    wordclass[单词名称][类别]表示单词在本类别中出现次数
    wordtotal[单词名称]表示包含它的文档数
    对于每个单词需要计算    它在本类别中出现次数  总文档中包含它的文档次数
    对于整体需要计算  每个类别下的单词数  总类别的文档总数
    """
    # 读取上一步保存的数据
    with open("./split_data.pkl", "rb") as f:
        labels = pickle.load(f)
        texts = pickle.load(f)
    # 划分训练集和测试集，大小各一半
    print(len(np.unique(labels)))
    trainText = []
    for i in range(len(labels)):
        trainText.append(labels[i] + ' ' + texts[i])
    # 数据随机
    random.shuffle(trainText)
    num = len(trainText)
    testText = trainText[num // 2:]
    trainText = trainText[:num // 2]
    print("训练集大小：", len(trainText))
    print("测试集大小：", len(testText))
    classes = ['it', 'auto', 'stock', 'yule', 'sports', 'business', 'health', 'learning', 'women', 'house']

    docCount = [0] * len(classes)  # 各类别单词计数
    wordCount = defaultdict(doc_dict)  # 每个单词在每个类别中的计数
    wordTotal = defaultdict(int) #每个单词出现的文档数
    docsum = len(trainText) #文档总数
    wordAll = set()
    # 扫描文件和计数
    for line in trainText:
        lable, text = line.split(' ', 1)
        index = lable2id(lable)  # 类别索引
        words = text.split(' ')
        for word in words:
            if word in [' ', '', '\n']:
                continue
            wordCount[word][index] += 1
            docCount[index] += 1
        wordunique = np.unique(words)
        for wordu in wordunique:
            wordTotal[wordu] += 1
            wordAll.add(wordu)

    # 计算每个词的TF值
    word_tf = defaultdict(doc_dict)  # 存储每个词的tf值
    for k,vs in wordCount.items():
        word_idf = math.log(docsum/(wordTotal[k]+1))
        for i in range(len(vs)):
            #print(wordCount[k][i])
            word_tf[k][i] = (wordCount[k][i] / docCount[i])*word_idf
    #miDict = defaultdict(doc_dict)

    fWords = set()
    #遍历每个单词
    # [('系列',
    #   [0.07069207490052264, 0.0004569980602163192, 0.00014678108673421724, 0.0004949781437722148, 8.229827417144861e-05,
    #    0.00019064948048112265, 1.7794987217198647e-05, 9.915790076623034e-05, 0.011894693491561964,
    #    1.651785279443384e-05]),]
    for i in range(len(docCount)):
        keyf = lambda x:x[1][i]
        sortedDict = sorted(word_tf.items(),key = keyf,reverse=True)
        t = ','.join([w[0] for w in sortedDict[:20]])
        for j in range(1000):
             fWords.add(sortedDict[j][0])
    out = open("ahah", 'w', encoding='utf-8')
    # 输出各个类的单词数目
    out.write(str(docCount) + "\n")
    # 输出tf-idf最高的词作为特征词
    for fword in fWords:
        out.write(fword + "\n")
    print("特征词写入完毕！")
    out.close()
    return testText, trainText,fWords
    # out = open("haha", 'w', encoding='utf-8')
    # for aa,bb in word_tf.items():
    #     out.write(aa+"\t"+str(bb)+"\n")
    # out.close()
def huxinxi():
    # 读取上一步保存的数据
    with open("./split_data.pkl", "rb") as f:
        labels = pickle.load(f)
        texts = pickle.load(f)
    # 划分训练集和测试集，大小各一半
    print(len(np.unique(labels)))
    trainText = []
    for i in range(len(labels)):
        trainText.append(labels[i] + ' ' + texts[i])
    # 数据随机
    random.shuffle(trainText)
    num = len(trainText)
    testText = trainText[num // 2:]
    trainText = trainText[:num // 2]

    print("训练集大小：", len(trainText))  # 685111
    print("测试集大小：", len(testText))  # 685112
    # 文章类别列表
    classes = ['it', 'auto', 'stock', 'yule', 'sports', 'business', 'health', 'learning', 'women', 'house']
    # 构造和类别数等长的0向量
    sys.exit(0)
    # 计算互信息，这里log的底取为2
    def mutual_info(N, Nij, Ni_, N_j):
        return 1.0 * Nij / N * math.log(N * 1.0 * (Nij + 1) / (Ni_ * N_j)) / math.log(2)

    # 统计每个词在每个类别出现的次数，和每类的文档数，计算互信息，提取特征词
    def count_for_cates(trainText, featureFile):
        docCount = [0] * len(classes)  # 各类别单词计数
        wordCount = defaultdict(doc_dict)  # 每个单词在每个类别中的计数

        # 扫描文件和计数
        for line in trainText:
            lable, text = line.strip().split(' ', 1)
            index = lable2id(lable)  # 类别索引
            words = text.split(' ')
            for word in words:
                if word in [' ', '', '\n']:
                    continue
                wordCount[word][index] += 1
                docCount[index] += 1

        # 计算互信息值
        print("计算互信息，提取特征词中，请稍后...")
        miDict = defaultdict(doc_dict)
        N = sum(docCount)

        # 遍历每个分类，计算词项k与文档类别i的互信息MI
        for k, vs in wordCount.items():
            for i in range(len(vs)):
                N11 = vs[i]  # 类别i下单词k的数量
                N10 = sum(vs) - N11  # 非类别i下单词k的数量
                N01 = docCount[i] - N11  # 类别i下其他单词数量
                N00 = N - N11 - N10 - N01  # 其他类别中非k单词数目
                mi = mutual_info(N, N11, N10 + N11, N01 + N11) + mutual_info(N, N10, N10 + N11,N00 + N10) + mutual_info(N, N01, N01 + N11,
                                                                                            N01 + N00) + mutual_info(
                    N, N00, N00 + N10, N00 + N01)
                miDict[k][i] = mi

        fWords = set()
        # 遍历每个单词
        for i in range(len(docCount)):
            keyf = lambda x: x[1][i]
            sortedDict = sorted(miDict.items(), key=keyf, reverse=True)
            # 打印每个类别中排名前20的特征词
            t = ','.join([w[0] for w in sortedDict[:20]])
            print(classes[i], ':', t)
            for j in range(1000):
                fWords.add(sortedDict[j][0])

        out = open(featureFile, 'w', encoding='utf-8')
        # 输出各个类的文档数目
        out.write(str(docCount) + "\n")
        # 输出互信息最高的词作为特征词
        for fword in fWords:
            out.write(fword + "\n")
        print("特征词写入完毕！")
        out.close()
    count_for_cates(trainText, 'featureFile')
    return testText,trainText
    # 从特征文件导入特征词
def load_feature_words(featureFile):
        f = open(featureFile, encoding='utf-8')
        #读取第一行
        docCounts = eval(f.readline())
        features = set()
        # 读取特征词
        for line in f:
            features.add(line.strip())
        f.close()
        # print("-----------")
        # print(docCounts+"!!!!!!!!!!\n"+features)
        # print("---------")
        # sys.exit(0)
        return docCounts, features
def modeltrain(t):
    # 训练贝叶斯模型，实际上计算每个类中特征词的出现次数
    #即类别i下单词k的出现概率，并使用拉普拉斯平滑（加一平滑）
    #ahah.txt,traintext,modelfile(空)
    def train_bayes(featureFile, textFile, modelFile):
        print("使用朴素贝叶斯训练中...")
        start = datetime.datetime.now()
        #docCounts为每一类单词总数，feature是词袋
        docCounts, features = load_feature_words(featureFile)  # 读取词频统计和特征词
        wordCount = defaultdict(doc_dict)

        #类别index中单词总数计数
        tCount = [0] * len(docCounts)
        # 遍历每个文档
        for line in textFile:
            lable, text = line.split(' ', 1)
            index = lable2id(lable)
            words = text.strip().split(' ')
            for word in words:
                if word in features and word not in [' ', '', '\n']:
                    tCount[index] += 1  # 类别index中单词总数计数
                    wordCount[word][index] += 1  # 类别index中单词word的计数
        end = datetime.datetime.now()
        print("训练完毕，写入模型...")
        print("程序运行时间：" + str((end - start).seconds) + "秒")

        # 加一平滑
        outModel = open(modelFile, 'w', encoding='utf-8')
        # 遍历每个单词
        for k, v in wordCount.items():
            # 遍历每个类别i，计算该类别下单词的出现概率（频率）
            scores = [(v[i] + 1) * 1.0 / (tCount[i] + len(wordCount)) for i in range(len(v))]
            outModel.write(k + "\t" + str(scores) + "\n")  # 保存模型，记录类别i下单词k的出现概率（频率）
        outModel.close()

    train_bayes('./ahah', t, './modelFile')
    # 从模型文件中导入计算好的贝叶斯模型
def load_model(modelFile):
        print("加载模型中...")
        f = open(modelFile, encoding='utf-8')
        scores = {}
        for line in f:
            word, counts = line.strip().rsplit('\t', 1)
            scores[word] = eval(counts)
        f.close()
        return scores

    # 预测文档分类，标准输入每一行为一个文档
def predict(featureFile, modelFile, testText,wa):
        docCounts, features = load_feature_words(featureFile)  # 读取词频统计和特征词
        #每类别的概率
        docScores = [math.log(count * 1.0 / sum(docCounts)) for count in docCounts]  # 每个类别出现的概率
        scores = load_model(modelFile)  # 加载模型，每个单词在类别中出现的概率
        indexList = []
        pIndexList = []
        start = datetime.datetime.now()
        print("正在使用测试数据验证模型效果...")
        for line in testText:
            lable, text = line.split(' ', 1)
            index = lable2id(lable)
            words = text.split(' ')
            preValues = list(docScores)
            for word in words:
                if word in wa and word not in [' ', '', '\n']:
                    for i in range(len(preValues)):
                        # 利用贝叶斯公式计算对数概率，后半部分为每个类别中单词word的出现概率
                        preValues[i] += math.log(scores[word][i])
            m = max(preValues)  # 取出最大值
            pIndex = preValues.index(m)  # 取出最大值类别的索引
            indexList.append(index) #测试文档原本分类
            pIndexList.append(pIndex) #预测分类

        end = datetime.datetime.now()
        print("程序运行时间：" + str((end - start).seconds) + "秒")
        return indexList, pIndexList
def doc_dict():
    classes = ['it', 'auto', 'stock', 'yule', 'sports', 'business', 'health', 'learning', 'women', 'house']
    return [0]*len(classes)
if __name__ == '__main__':
    #join阻塞
    #get_classify_count("./cor.txt")
    #get_stopword()
    ta,t,wa = tfidf()
    modeltrain(t)
    indexList, pIndexList = predict('./featureFile', './modelFile', ta,wa)
    C=confusion_matrix(indexList, pIndexList)
    classes = ['it', 'auto', 'stock', 'yule', 'sports', 'business', 'health', 'learning', 'women', 'house']
    pd.DataFrame(C, index=classes, columns=classes)
    #计算各类的精确率，召回率，F1值
    p = precision_score(indexList, pIndexList, average=None)
    r = recall_score(indexList, pIndexList, average=None)
    f1 = f1_score(indexList, pIndexList, average=None)
    p_max,r_max,p_min,r_min = 0,0,1,1
    for i in range(len(classes)):
      print("类别{:8}".format(classes[i]), end=" ")
      print("精确率为：{}, 召回率为：{}, F1值为：{}".format(p[i], r[i], f1[i]))
      if (p[i] > p_max): p_max = p[i]
      if (r[i] > r_max): r_max = r[i]
      if (p[i] < p_min): p_min = p[i]
      if (r[i] < r_min): r_min = r[i]

    #计算总体的精确率，召回率，F1值
    pa = precision_score(indexList, pIndexList, average="micro")
    ra = recall_score(indexList, pIndexList, average="micro")
    f1a = f1_score(indexList, pIndexList, average="micro")
    print("总体{:8}".format("====>"), end=" ")
    print("精确率为：{}, 召回率为：{}, F1值为：{}".format(pa, ra, f1a))
    print("最大{:8}".format("====>"), end=" ")
    print("精确率为：{}, 召回率为：{}".format(p_max, r_max))
    print("最小{:8}".format("====>"), end=" ")
    print("精确率为：{}, 召回率为：{}".format(p_min, r_min))

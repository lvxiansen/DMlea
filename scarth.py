import pandas as pd
import numpy as np
import jieba
import jieba.analyse
import gensim
from gensim import corpora,models,similarities
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# https://www.cnblogs.com/xiaoyh/p/11453364.html
# 读取训练集
# read_table()从文件、url、文件型对象中加载带分隔符的数据，默认为'\t'
# 在实际使用中可以通过对sep参数的控制来对任何文本文件读取
df_news = pd.read_table('./val.txt',names=['category','theme','URL','content'],encoding='utf-8',nrows=2000)
df_news = df_news.dropna() # 删除缺失数据
#df_news.head().URL # 读取前几行数据，默认是5.怎么只读theme？df_news.head().theme
#df_news.shape #返回维度


# 训练集列表化并转化为DataFrame格式
content = df_news.content.values.tolist() # 转换为list 实际上是二维list
content_S = []
for line in content:
    # jieba分词 精确模式。返回一个列表类型，建议使用
    current_segment = jieba.lcut(line)
    if len(current_segment) > 1 and current_segment != '\r\n':
        content_S.append(current_segment)
df_content = pd.DataFrame({'content_S':content_S})   # 转换为DataFrame
#print(df_content.head())


# 读取停词表
stopwords = pd.read_csv('./stopwords.txt',index_col=False,sep='\t',quoting=3,names=['stopword'],encoding='utf-8')
#print(stopwords.head())


# 删除新闻中的停用词
def drop_stopwords(contents, stopwords):
    contents_clean = []  # 删除后的新闻
    all_words = []  # 构造词云所用的数据
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
            all_words.append(str(word))
        contents_clean.append(line_clean)
    return contents_clean, all_words
contents = df_content.content_S.values.tolist()
stopwords = stopwords.stopword.values.tolist()


# 得到删除停用词后的新闻以及词云数据
contents_clean, all_words = drop_stopwords(contents, stopwords)


df_content = pd.DataFrame({'contents_clean':contents_clean})
#print(df_content.head())


index = 20
#print(df_news['content'][index]) #df_news是dataFramed格式
content_S_str = ''.join(content_S[index]) #content_S是list格式
# print(content_S_str)
# 提取关键词
#print("  ".join(jieba.analyse.extract_tags(content_S_str, topK=5, withWeight=False)))


# 进行词映射，相当于一个大的字典，每一个词汇进行一个映射。
# 做映射，相当于词袋 格式要求：list of list
dictionary = corpora.Dictionary(contents_clean) # 字典
corpus = [dictionary.doc2bow(sentence) for sentence in contents_clean] # 语料

# num_topics=20 类似Kmeans自己指定K值
# 进行LDA建模，将整个语料库划分为20个主题
lda = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=dictionary,num_topics=20)

# 一号分类结果
print(lda.print_topic(1, topn=5))

for topic in lda.print_topics(num_topics=20,num_words=5):
    print(topic[1])

df_train=pd.DataFrame({'contents_clean':contents_clean,'label':df_news['category']})
df_train.tail()
df_train.label.unique()

label_mapping = {"汽车": 1, "财经": 2, "科技": 3, "健康": 4, "体育":5, "教育": 6,"文化": 7,"军事": 8,"娱乐": 9,"时尚": 0}
df_train['label'] = df_train['label'].map(label_mapping)
print(df_train.head())

x_train, x_test, y_train, y_test = train_test_split(df_train['contents_clean'].values, df_train['label'].values, random_state=1)
#x_train = x_train.flatten()
#x_train[0][1]

words = []
for line_index in range(len(x_train)):
    try:
        #x_train[line_index][word_index] = str(x_train[line_index][word_index])
        words.append(' '.join(x_train[line_index]))
    except:
        print(line_index,word_index)
#print(words[0])

vec = CountVectorizer(analyzer='word', max_features=4000,  lowercase = False)
vec.fit(words)
classifier = MultinomialNB()
classifier.fit(vec.transform(words), y_train)
test_words = []
for line_index in range(len(x_test)):
    try:
        #x_train[line_index][word_index] = str(x_train[line_index][word_index])
        test_words.append(' '.join(x_test[line_index]))
    except:
         print (line_index,word_index)
print(test_words[0])
print(classifier.score(vec.transform(test_words), y_test))
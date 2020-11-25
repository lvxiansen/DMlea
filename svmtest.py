from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
train_data = fetch_20newsgroups(subset='train', shuffle=True,categories=categories,random_state=10)

#特征向量化&TF-IDF&标准化
tfidf_transformer = TfidfVectorizer()
tf_train_data = tfidf_transformer.fit_transform(["这是 一本 好书","故宫 很 美好"])

#SVM模型训练及预测
clf = SVC(kernel = 'linear').fit(tf_train_data, train_data.target)
docs_new = ['God is love', 'OpenGL on the GPU is fast']
tf_docs_new = tfidf_transformer.transform(docs_new)
predicted = clf.predict(tf_docs_new)
for doc, category in zip(docs_new, predicted):
    print(doc + '>>' + train_data.target_names[category])

test_data = fetch_20newsgroups(subset='test', shuffle=True, categories=categories,random_state=23)
tf_test_data = tfidf_transformer.transform("")
predicted = clf.predict(tf_test_data)
print("训练集评分:", clf.score(tf_train_data, train_data.target))
print("测试集评分", clf.score(tf_test_data, test_data.target))

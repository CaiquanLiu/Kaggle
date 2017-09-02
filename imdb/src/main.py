# coding:utf-8
'''
代码来源《Python机器学习及实践-从零开始通往Kaggle竞赛之路》
Created on 2017/8/30 上午9:26

@author: liucaiquan
'''
import pandas as pd

#############################
# 数据读入
train = pd.read_csv('../datasets/labeledTrainData.tsv', delimiter='\t')
test = pd.read_csv('../datasets/testData.tsv', delimiter='\t')
# print type(train)
# print train.head()

from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords


# 数据预处理(清洗)
def review_to_text(review, remove_stopwords):
    # 去除html标记
    raw_text = BeautifulSoup(review, 'lxml').get_text()

    # 去除非字母字符
    letters = re.sub('[^a-zA-Z]', ' ', raw_text)
    # print letters
    words = letters.lower().split()

    # 去除停用词
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]
    return words


# review = train['review'][0]
# print review
# print review_to_text(review, True)

# 训练集
X_train = []
for review in train['review']:
    X_train.append(' '.join(review_to_text(review, True)))
print X_train
# print 'test-pause'
# raw_input()

# 测试集
X_test = []
for review in test['review']:
    X_test.append(' '.join(review_to_text(review, True)))

# 训练集分类结果
y_train = train['sentiment']

#######################################
# 基于朴素贝叶斯(MultinominalNB)模型的分类器，使用CountVectorizer和TfidVectorizer对文本特征进行抽取
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

# 流水线
pip_count = Pipeline([('count_vec', CountVectorizer(analyzer='word')), ('mnb', MultinomialNB())])
pip_tfidf = Pipeline([('tfidf_vec', TfidfVectorizer(analyzer='word')), ('mnb', MultinomialNB())])

# 超参数搜索
params_count = {'count_vec__binary': [True, False], 'count_vec__ngram_range': [(1, 1), (1, 2)],
                'mnb__alpha': [0.1, 1.0, 10.0]}
params_tfidf = {'tfidf_vec__binary': [True, False], 'tfidf_vec__ngram_range': [(1, 1), (1, 2)],
                'mnb__alpha': [0.1, 1.0, 10.0]}
gs_count = GridSearchCV(pip_count, params_count, cv=4, n_jobs=-1, verbose=1)

# 训练
gs_count.fit(X_train, y_train)
print gs_count.best_score_
print gs_count.best_params_

# 分类
count_y_predict = gs_count.predict(X_test)

# 超参数搜索
gs_tfidf = GridSearchCV(pip_tfidf, params_tfidf, cv=4, n_jobs=-1, verbose=1)

# 训练
gs_tfidf.fit(X_train, y_train)
print gs_tfidf.best_score_
print gs_tfidf.best_params_

# 预测
tfidf_y_predict = gs_tfidf.predict(X_test)

# 结果保存
submission_count = pd.DataFrame({'id': test['id'], 'sentiment': count_y_predict})
submission_tfidf = pd.DataFrame({'id': test['id'], 'sentiment': tfidf_y_predict})
submission_count.to_csv('../rst/submission_count.csv', index=False)
submission_tfidf.to_csv('../rst/submission_tfidf.csv', index=False)

########################
# # 使用未标记数据进word2vec训练，采用GradientBoostingClassifier对影评进行训练和分类（测试未通过）
# # 读入未标记数据
# unlabeled_train = pd.read_csv('../datasets/unlabeledTrainData.tsv', delimiter='\t', quoting=3)
#
# import nltk.data
#
# tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
#
#
# # print tokenizer
#
# #
# def review_to_sentences(review, tokenizer):
#     raw_sentences = tokenizer.tokenize(review.strip())
#     sentences = []
#     for raw_sentence in raw_sentences:
#         if len(raw_sentence) > 0:
#             sentences.append(review_to_text(raw_sentence, False))
#     return sentences
#
#
# corpora = []
# for review in unlabeled_train['review']:
#     # print review
#     corpora += review_to_sentences(review.decode('utf8'), tokenizer)
#     # print '=================='
#     # print corpora
#
# num_features = 300
# min_word_count = 20
# num_workers = 4
# context = 10
# downsampling = 1e-3
#
# # word2vec模型训练
# from gensim.models import word2vec
#
# model = word2vec.Word2Vec(corpora, workers=num_workers, size=num_features, min_count=min_word_count, window=context,
#                           sample=downsampling)
# model.init_sims(replace=True)
# model_name = '../datasets/300features_20minwords_10context'
# model.save(model_name)
#
# # from gensim.models import Word2Vec
# # model=Word2Vec.load(model_name)
# # print model.most_similar('man')
#
# import numpy as np
#
#
# #
# def makeFeatureVec(words, model, num_features):
#     featureVec = np.zeros((num_features,), dtype="float32")
#     nwords = 0
#     index2word_set = set(model.wv.index2word)
#     for word in words:
#         if word in index2word_set:
#             nwords = nwords + 1.
#             featureVec = np.add(featureVec, model[word])
#     featureVec = np.divide(featureVec, nwords)
#     return featureVec
#
#
# #
# def getAvgFeatureVecs(reviews, model, num_features):
#     counter = 0
#     reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
#
#     for review in reviews:
#         reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
#         counter += 1
#     return reviewFeatureVecs
#
#
# # 训练集特征
# clean_train_reviews = []
# for review in train['review']:
#     clean_train_reviews.append(review_to_text(review, remove_stopwords=True))
# trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)
#
# # 测试集特征
# clean_test_reviews = []
# for review in test['review']:
#     clean_test_reviews.append(review_to_text(review, remove_stopwords=True))
# testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)
#
# # 使用GradientBoostingClassifier对影评进行训练和分类
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.grid_search import GridSearchCV
#
# gbc = GradientBoostingClassifier()
#
# # 超参数搜索
# params_gbc = {'n_estimators': [10, 100, 500], 'learning_rate': [0.01, 0.1, 1.0], 'max_depth': [2, 3, 4]}
# gs = GridSearchCV(gbc, params_gbc, cv=4, n_jobs=-1, verbose=1)
#
# # 训练
# gs.fit(trainDataVecs, y_train)
# print gs.best_score_
# print gs.best_params_
#
# # 分类
# result = gs.predict(testDataVecs)
#
# # 结果保存
# output = pd.DataFrame(data={'id': test['id'], 'sentiment': result})
# output.to_csv('../rst/submission_w2v.csv', index=False, quoting=3)

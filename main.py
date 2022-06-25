import jieba as jieba
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
# 保存模型
output_dir = 'output'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

data = pd.read_excel('E:\整合\合并.xls').astype(str)
stopwords = set()
with open('stopwords.txt',encoding='utf-8') as infile:
    for line in infile:
        line = line.rstrip('\n')
        if line:
            stopwords.add(line.lower())
from sklearn.feature_extraction.text import  TfidfVectorizer,CountVectorizer
tfidf= TfidfVectorizer(tokenizer=jieba.lcut,stop_words=stopwords, max_df=0.3,min_df=50)
x = tfidf.fit_transform(data['内容'].values.astype(str))
x_train, x_test, y_train, y_test = train_test_split(data['内容'],data['标签'], random_state=1)
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
classifier = MultinomialNB(alpha=0.001)
#模型训练
classifier.fit(tfidf.transform(x_train), y_train)
print(classifier.score(tfidf.transform(x_test), y_test))
# 保存模型
import dill
model_file = os.path.join(output_dir,'model.pkl')
with open(model_file,'wb') as outfile:
    dill.dump({
        'tfidf':tfidf,
        'lr':classifier
    },outfile)

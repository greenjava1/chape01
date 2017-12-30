#coding:utf-8
import os
import sys

import scipy as sp

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=1,stop_words='english')
DIR = r"../data/toy"
print(os.listdir(DIR))
posts = [open(os.path.join(DIR, f)).read() for f in os.listdir(DIR)]  # 注意，os.listdir, 读取文档
print(posts)


#print (vectorizer)
#content = ['How to format my hard disk','Hard disk format problems']
#X=vectorizer.fit_transform(content)
#print (vectorizer.get_feature_names())
#print(X)
#print (X.toarray().transpose())

X_train = vectorizer.fit_transform(posts)
num_samples,num_features = X_train.shape
print (num_samples,num_features)
#print(X_train)
#print(vectorizer.get_feature_names())

new_post = "imaging databases"
new_post_vec = vectorizer.transform([new_post])
#print(new_post_vec)
#print(new_post_vec.toarray())

import scipy as sp
def dist_raw(v1,v2):

    delta = v1-v2
    print(v1.toarray())
    print('==================')
    print(v2.toarray())
    return sp.linalg.norm(delta.toarray())

def dist_norm(v1,v2):
    v1_normalized = v1/sp.linalg.norm(v1.toarray())
    v2_normalized = v2/sp.linalg.norm(v2.toarray())
    delta = v1_normalized - v2_normalized
    return sp.linalg.norm(delta.toarray())

import sys
best_doc = None
best_dist = 10000000000000
best_i = None
for i in range(0, num_samples):
    post = posts[i]
    if post == new_post:
        continue
    post_vec = X_train.getrow(i)
    d = dist_norm(post_vec, new_post_vec)
    print  ("== post %i with dist=%.2f: %s" % (i, d, post))
    if d < best_dist:
        best_dist = d
        best_i = i
print("Best post is %i with dist=%.2f" % (best_i, best_dist))
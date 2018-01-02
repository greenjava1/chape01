#coding:utf-8
import os
import sys
import scipy as sp
# 下面装载的就是计算单词词频，然后表示成向量的类
from sklearn.feature_extraction.text import CountVectorizer

# 可以help（CountVectorizer）查看内部的参数说明
DIR = r"../data/toy"
posts = [open(os.path.join(DIR, f)).read() for f in os.listdir(DIR)]
# 上面就是用for搜索toy文件夹下不同的文件，然后逐个打开，并将每个文件的内容存
# 储成一个列表的一个元素
new_post = "imaging databases"  # 用户询问的语句

import nltk.stem  # c）装载nltk的stem

english_stemmer = nltk.stem.SnowballStemmer('english')  # c）提取‘english’集合的stemmer


# c）例如输入english_stemer.stem（‘imaging’）返回的就是u’imag’，即相同意思不同表示

##########用nltk的stemmer扩展计算词频向量函数############
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()  # c）提取计算词频类中方法？
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))  # c）通过nltk逐个处理单词


# vectorizer = CountVectorizer(min_df=1, stop_words='english',
# preprocessor=stemmer)
# b）传递需要的参数：忽略低于出现1次的单词;采用‘english’集合的 stop word
vectorizer = StemmedCountVectorizer(min_df=1, stop_words='english')
# b）c）得到装有nltk的词频向量类扩展版
##########################################################

# 上面是基于CountVectorizer的；下面是基于TfidfVectorizer的

##########用nltk的stemmer扩展计算tf-idf的词频向量函数############
from sklearn.feature_extraction.text import TfidfVectorizer


# d）装载sklearn中的tfidf向量类，而且该TfidfVectorize是继承自CountVectorizer类

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


vectorizer = StemmedTfidfVectorizer(min_df=1, stop_words='english')
# d）得到加入nltk的tfidf版本的计算词频向量函数
##########################################################

print(vectorizer)
# 使用这个类计算得到的就不再是词频，而是TF-IDF值

X_train = vectorizer.fit_transform(posts)
# 将传递的参数表示的列表转换成对应的特征向量，书p53

num_samples, num_features = X_train.shape
# 样本个数：特征词个数（字典的维度）
print("#samples: %d, #features: %d" % (num_samples, num_features))

new_post_vec = vectorizer.transform([new_post])
# 将用户询问文本转换成词频（或者TF-IDF）向量
print(new_post_vec, type(new_post_vec))
# 默认以稀疏矩阵存储
print(new_post_vec.toarray())
# 呈现全矩阵形式，将其中0值也显示出来
print(vectorizer.get_feature_names())


# 显示所包含的特征词，即形成字典中每个词的名称

#################计算两个词频向量之间的距离######################
def dist_raw(v1, v2):
    delta = v1 - v2
    return sp.linalg.norm(delta.toarray())
    # a）返回2范数的结果，因为计算范数的时候需要全矩阵形式


def dist_norm(v1, v2):
    v1_normalized = v1 / sp.linalg.norm(v1.toarray())
    v2_normalized = v2 / sp.linalg.norm(v2.toarray())
    # a）标准化了每个单独的词频向量，解决文本中故意重复的问题
    # a）比如原本是‘abc’，故意重复就是‘abcabcabc’
    delta = v1_normalized - v2_normalized
    return sp.linalg.norm(delta.toarray())


dist = dist_norm
# 简化函数名称
################################################################

best_dist = sys.maxsize
# 将系统中的最大值作为向量之间的初始距离（类似于无穷大）
best_i = None

##########对每个样本进行处理#####################
for i in range(0, num_samples):
    post = posts[i]
# 读取当前需要处理的原始文本，以便显示
    if post == new_post:
        continue
    # 如果与询问的相同，则跳过计算
    post_vec = X_train.getrow(i)
# 读取第I 行的词频向量
    d = dist(post_vec, new_post_vec)
# 计算两个向量的欧式距离
    print("=== Post %i with dist=%.2f: %s" % (i, d, post))

    if d < best_dist:
        best_dist = d
        best_i = i
    # 记录最好的匹配位置的距离和索引
print("Best post is %i with dist=%.2f" % (best_i, best_dist))
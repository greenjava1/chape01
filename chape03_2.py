#coding:utf-8
#http://qwone.com/~jason/20Newsgroups/

import sklearn.datasets
import scipy as sp
MLCOMP_DIR = r"D:\machine_learn\data\20news-18828"
groups = [
    'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware', 'comp.windows.x', 'sci.space']

groups1 = [
    'comp.graphics','comp.os.ms-windows.misc']

#dataset = sklearn.datasets.load_mlcomp("20news-18828", "train",mlcomp_root=None, groups)
dataset = sklearn.datasets.load_files(MLCOMP_DIR, "train",groups)
print("Number of posts:", len(dataset.filenames))

labels = dataset.target
num_clusters = 50  # sp.unique(labels).shape[0]

import nltk.stem

english_stemmer = nltk.stem.SnowballStemmer('english')

from sklearn.feature_extraction.text import TfidfVectorizer

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

vectorizer = StemmedTfidfVectorizer(min_df=10, max_df=0.5,
                                    # max_features=1000,
                                    stop_words='english', decode_error='replace'
                                    )

vectorized = vectorizer.fit_transform(dataset.data)
num_samples, num_features = vectorized.shape
print("#samples: %d, #features: %d" % (num_samples, num_features))

from sklearn.cluster import KMeans

km = KMeans(n_clusters=num_clusters, init='k-means++', n_init=1,
            verbose=1)

clustered = km.fit(vectorized)

from sklearn import metrics  # 许多评估函数在此包中

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand Index: %0.3f" %
      metrics.adjusted_rand_score(labels, km.labels_))
print("Adjusted Mutual Information: %0.3f" %
      metrics.adjusted_mutual_info_score(labels, km.labels_))
print(("Silhouette Coefficient: %0.3f" %
       metrics.silhouette_score(vectorized, labels, sample_size=1000)))


new_post = \
    """Disk drive problems. Hi, I have a problem with my hard disk. 
After 1 year it is working only sporadically now. 
I tried to format it, but now it doesn't boot any more. 
Any ideas? Thanks. 
"""

new_post_vec = vectorizer.transform([new_post])
new_post_label = km.predict(new_post_vec)[0]

similar_indices = (km.labels_ == new_post_label).nonzero()[0]

print (new_post_label)
print (km.labels_)

similar = []
for i in similar_indices:
    dist = sp.linalg.norm((new_post_vec - vectorized[i]).toarray())
    similar.append((dist, dataset.data[i]))

similar = sorted(similar)

show_at_1 = similar[0]
show_at_2 = similar[int(len(similar) / 2)]
show_at_3 = similar[-1]

print(show_at_1)
print(show_at_2)
print(show_at_3)

import pdb

pdb.set_trace()
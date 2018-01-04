#coding:utf-8
#nltk词干提取
from nltk import SnowballStemmer
import nltk.stem
import math
import scipy as sp
s = nltk.stem.SnowballStemmer('english')
#print(s.stem('booked'))
#print(SnowballStemmer.languages)

import gensim.corpora.wikicorpus
wiki = gensim.corpora.wikicorpus.WikiCorpus('simplewiki-20171220-pages-articles-multistream.xml.bz2')
wiki.dictionary.save_as_text('wiki_en_wordids.txt')

def tfidf(term,doc,docset):
    print ('********************************************')
    tf = float(doc.count(term))/sum(doc.count(term) for  doc in docset)
    print ('doc.count(term)=',(doc.count(term)))
    print ('sum(doc.count(term) for  doc in docset)=',sum(doc.count(term) for  doc in docset))

    idf = math.log(float(len(docset))/(len([doc for doc in docset if term in doc])))
    print ('fenmu=',(len([doc for doc in docset if term in doc])))
    print ('idf=',idf)
    print ('============================================')
    return tf*idf

a,abb,abc = ['a'],['a','b','b'],['a','b','c']
D=[a,abb,abc]
print (tfidf('a',a,D))
print (tfidf('b',abb,D))
print (tfidf('a',abc,D))
print (tfidf('b',abc,D))
print (tfidf('c',abc,D))




#chapte4


warned_of_error = False

def create_cloud(oname, words,maxsize=120, fontname='Lobster'):
    '''Creates a word cloud (when pytagcloud is installed)

    Parameters
    ----------
    oname : output filename
    words : list of (value,str)
    maxsize : int, optional
        Size of maximum word. The best setting for this parameter will often
        require some manual tuning for each input.
    fontname : str, optional
        Font to use.
    '''
    try:
        from pytagcloud import create_tag_image, make_tags
    except ImportError:
        if not warned_of_error:
            print("Could not import pytagcloud. Skipping cloud generation")
        return

    # gensim returns a weight between 0 and 1 for each word, while pytagcloud
    # expects an integer word count. So, we multiply by a large number and
    # round. For a visualization this is an adequate approximation.
    words = [(w,int(v*10000)) for w,v in words]
    tags = make_tags(words, maxsize=maxsize)
    create_tag_image(tags, oname, size=(1800, 1200), fontname=fontname)

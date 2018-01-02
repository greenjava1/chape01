#coding:utf-8
#nltk词干提取
from nltk import SnowballStemmer
import nltk.stem
import math
import scipy as sp
s = nltk.stem.SnowballStemmer('english')
#print(s.stem('booked'))
#print(SnowballStemmer.languages)

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
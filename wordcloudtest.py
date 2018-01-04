#coding='utf-8'
"""
@author:zj
@software:PyCharm
@time:2017/6/30
"""
from os import path
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator
import pickle
import jieba
import codecs
text = ''
with open('记忆与印象.txt','r',encoding='gbk') as fin:
    for line in fin.readlines():
        line = line.strip('\n')
        text += ' '.join(jieba.cut(line))
        text += ' '
        fout = open('text.txt','wb')
        pickle.dump(text,fout)
        fout.close()
fr=open('text.txt','rb')
text=pickle.load(fr)
print("加载成功")
backgroud_Image=plt.imread('alice.jpg')
print('加载图片成功！')
'''设置词云样式'''
wc=WordCloud(
background_color='white',
mask=backgroud_Image,
font_path='C:\Windows\Fonts\STZHONGS.TTF', #若是有中文的话，这句代码必须添加，不然会出现方框，不出现汉字
max_words=2000,
stopwords=STOPWORDS,
max_font_size=150,
random_state=30
)
wc.generate_from_text(text)
print('开始加载文本')
img_colors=ImageColorGenerator(backgroud_Image)
wc.recolor(color_func=img_colors)
plt.imshow(wc)
plt.axis('off')
plt.show()
print('display success!')
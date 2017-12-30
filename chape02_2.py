#coding:utf-8
from load import load_dataset


def test_iris():
    features, labels = load_dataset('iris')
    assert len(features[0]) == 4  # 断言
    assert len(features)
    assert len(features) == len(labels)


def test_seeds():
    features, labels = load_dataset('seeds')
    assert len(features[0]) == 7
    assert len(features)
    assert len(features) == len(labels)


import milksets.iris
import milksets.seeds


def save_as_tsv(fname, module):
    features, labels = module.load()
    nlabels = [module.label_names[ell] for ell in labels]
    with open(fname, 'w') as ofile:
        for f, n in zip(features, nlabels):
            print >> ofile, "\t".join(map(str, f) + [n])


save_as_tsv('iris.tsv', milksets.iris)
save_as_tsv('seeds.tsv', milksets.seeds)

COLOUR_FIGURE = False

from matplotlib import pyplot as plt
from sklearn.datasets import load_iris

data = load_iris()
features = data['data']
feature_names = data['feature_names']
species = data['target_names'][data['target']]

setosa = (species == 'setosa')
features = features[~setosa]
species = species[~setosa]
virginica = species == 'virginica'

t = 1.75
p0, p1 = 3, 2

if COLOUR_FIGURE:
    area1c = (1., .8, .8)
    area2c = (.8, .8, 1.)
else:
    area1c = (1., 1, 1)
    area2c = (.7, .7, .7)

x0, x1 = [features[:, p0].min() * .9, features[:, p0].max() * 1.1]  # 注意用列表对两个变量赋值， 列表内为两个变量
y0, y1 = [features[:, p1].min() * .9, features[:, p1].max() * 1.1]

plt.fill_between([t, x1], [y0, y0], [y1, y1], color=area2c)  # 填充颜色
plt.fill_between([x0, t], [y0, y0], [y1, y1], color=area1c)
plt.plot([t, t], [y0, y1], 'k--', lw=2)
plt.plot([t - .1, t - .1], [y0, y1], 'k:', lw=2)
plt.scatter(features[virginica, p0], features[virginica, p1], c='b', marker='o')
plt.scatter(features[~virginica, p0], features[~virginica, p1], c='r', marker='x')
plt.ylim(y0, y1)
plt.xlim(x0, x1)
plt.xlabel(feature_names[p0])
plt.ylabel(feature_names[p1])
plt.savefig('../1400_02_02.png')

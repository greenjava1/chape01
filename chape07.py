# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

# This script shows an example of simple (ordinary) linear regression

# The first edition of the book NumPy functions only for this operation. See
# the file boston1numpy.py for that version.

'''
  :Attribute Information (in order):
      - CRIM     per capita crime rate by town
      - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
      - INDUS    proportion of non-retail business acres per town
      - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
      - NOX      nitric oxides concentration (parts per 10 million)
      - RM       average number of rooms per dwelling
      - AGE      proportion of owner-occupied units built prior to 1940
      - DIS      weighted distances to five Boston employment centres
      - RAD      index of accessibility to radial highways
      - TAX      full-value property-tax rate per $10,000
      - PTRATIO  pupil-teacher ratio by town
      - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
      - LSTAT    % lower status of the population
      - MEDV     Median value of owner-occupied homes in $1000's
'''

import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

boston = load_boston()
x = boston.data

y = boston.target

# Fitting a model is trivial: call the ``fit`` method in LinearRegression:
lr = LinearRegression()
print(type(x))
lr.fit(x, y)

# The instance member `residues_` contains the sum of the squared residues
rmse = np.sqrt(lr._residues/len(x))
print('RMSE: {}'.format(rmse))

fig, ax = plt.subplots()
# Plot a diagonal (for reference):
ax.plot([0, 50], [0, 50], '-', color=(.9,.3,.3), lw=4)

# Plot the prediction versus real:
ax.scatter(lr.predict(x), boston.target)

ax.set_xlabel('predicted')
ax.set_ylabel('real')
fig.savefig('Figure_07_08.png')
plt.show()
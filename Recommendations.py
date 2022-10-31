'''
genre key:
0 =
1 =
2 =
3 =
4 =
5 =
6 =

'''

import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df = pandas.read_csv("movies.csv")

d = {'UK': 0, 'USA': 1, 'N': 2}

features = ['year', 'genre', 'avg_vote', 'votes', 'go']

X = df[features]
y = df['go']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

tree.plot_tree(dtree, feature_names=features)
plt.show()

print(dtree.predict([[40, 10, 7, 1]]))
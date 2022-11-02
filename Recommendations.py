'''
genre key:
0 = Action
1 = Adventure
2 = Animation
3 = Biography
4 = Comedy
5 = Crime
6 = Drama
7 = Fantasy
8 = Family
9 = Film-Noir
10 = History
11 = Horror
12 = Musical
13 = Music
14 = Mystery
15 = Romance
16 = Sci-Fi
17 = Thriller
18 = War
19 = Western

'''

import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df = pandas.read_csv("movies.csv")


features = ['year', 'genre', 'avg_vote', 'votes']

X = df[features]
y = df['go']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

tree.plot_tree(dtree, feature_names=features)
plt.show()

print(dtree.predict([[40, 10, 7, 1]]))

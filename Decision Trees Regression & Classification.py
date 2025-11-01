from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
model = tree.DecisionTreeClassifier()
clf = model.fit(iris.data, iris.target)
import graphviz
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("iris_tree", format="pdf")
graph.view()

import sklearn.tree as sktree
sktree.export_graphviz(clf, out_file='tree.dot')
print (model.predict(iris.data[:5]))
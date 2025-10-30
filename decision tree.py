# Non-parametric supervised algorithm
# Used for classification & regression

# Q: Write a Python program to demonstrate how Decision Trees can be used for both classification and regression, using the Iris dataset. Visualize the tree using Graphviz and show predictions for sample data.

from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
model = tree.DecisionTreeClassifier()
mt = model.fit(iris.data, iris.target)

import graphviz #is a powerful visualization library used to draw graphs and trees, especially in machine learning for decision trees and model explainability.
dot_data = tree.export_graphviz(mt, out_file=None)
# exports a trained decision tree (like DecisionTreeClassifier or DecisionTreeRegressor) into a format that Graphviz can visualize.
#The DOT format is a simple graph description language used by Graphviz.

#Here are the most useful parameters (you can make the visualization much richer):

# dot_data = tree.export_graphviz(
#     clf,
#     out_file=None,                     # keep in memory instead of writing a .dot file
#     feature_names=iris.feature_names,  # column names for splits
#     class_names=iris.target_names,     # labels for classes
#     filled=True,                       # fill nodes with color
#     rounded=True,                      # rounded node boxes
#     special_characters=True            # allows special characters in names
# )

graph = graphviz.Source(dot_data)

import sklearn.tree as sktree
sktree.export_graphviz(mt, out_file='tree.dot')
# Writes the output into a .dot file on disk.
# Then you can visualize it later using Graphviz commands from the terminal or a GUI.
print(mt.predict(iris.data[:5]))

#out_file= option

# | Options                  | What it does               | When to use                                      |
# | :---------------------- | :------------------------- | :----------------------------------------------- |
# | `None`                  | Returns DOT string         | Interactive work, Jupyter, visualization in code |
# | `'tree.dot'`            | Writes file to disk        | When you want to manually view or share the tree |
# | `open('tree.dot', 'w')` | Writes file (with control) | When you need to manage file streams yourself    |
# | `io.StringIO()`         | Stores text in memory      | When integrating into pipelines or apps          |

#Decision Tree for Regression
from sklearn.tree import DecisionTreeRegressor
model = tree.DecisionTreeRegressor()
model.fit(iris.data[:,:3], iris.data[:,3])

print (model.predict(iris.data[:10,:3]))
print(iris.data[:10])

# 🌈 Visualize directly inside Python (Jupyter or script)
graph = graphviz.Source(dot_data)
graph.render("iris_tree", format="png")  # Saves as iris_tree.png
graph.view()  # <- In Jupyter, this displays the image inline

# Notes:

# render() just saves the file (PNG, PDF, etc.).
# view() saves and opens the file automatically.
# Make sure Graphviz executables (dot.exe) are in your system PATH.
# You won’t see it inline in VS Code, unlike Jupyter — you always open the saved file.


#if we use matplotlib

# import matplotlib.pyplot as plt
# from sklearn import tree

# plt.figure(figsize=(15,10))
# tree.plot_tree(
#     mt,
#     feature_names=iris.feature_names,
#     class_names=iris.target_names,
#     filled=True,
#     rounded=True
# )
# plt.show()
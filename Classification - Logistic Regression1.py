import numpy as np
def sigmoid(inputs):
    sigmoid_score = [1/float(1+np.exp(-x)) for x in inputs]
    return sigmoid_score

#model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

# load the data
from sklearn.datasets import load_breast_cancer
cancer_data = load_breast_cancer()

#train test
from sklearn.model_selection import train_test_split
xtr, xte, ytr, yte = train_test_split(cancer_data.data, cancer_data.target, test_size=0.2, random_state=42)


lr = LogisticRegression(
    C=1.0,
    class_weight=None,
    dual=False,
    fit_intercept=True,
    intercept_scaling=1,
    max_iter=100,
    multi_class='ovr',
    n_jobs=None,
    penalty='l2',
    random_state=None,
    solver='liblinear',
    tol=0.001,
    verbose=0,
    warm_start=False
)
lr.fit(xtr, ytr)
print(lr.score(xte, yte))

      
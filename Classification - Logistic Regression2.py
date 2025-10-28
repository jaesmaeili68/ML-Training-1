from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import numpy as np


def sigmoid(inputs):
    sigmoid_score = [1/float(1+np.exp(-x)) for x in inputs]
    return sigmoid_score


# model
lr = LogisticRegression()

# load the data
cancer_data = load_breast_cancer()
data = cancer_data['data']
target = cancer_data['target']

# train test
xtr, xte, ytr, yte = train_test_split(
    data, target, test_size=0.2, random_state=42)


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

print(lr.predict_proba(xtr[:10]))
print(lr.predict(xtr[:10]))

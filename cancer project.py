from sklearn.datasets import load_breast_cancer
bc = load_breast_cancer()

#dataset info
# print(bc.target.shape)
# print(bc.data.shape)

#preprocessing
from sklearn.model_selection import train_test_split
xtr, xte, ytr, yte = train_test_split(bc.data,bc.target, test_size=0.2)

print(f"features => train: {xtr.shape} - test: {xte.shape}")
print(f"label => train: {ytr.shape} - test: {yte.shape}")

#Normalize
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
xtr = scaler.fit_transform(xtr)
xte = scaler.transform(xte)

#classification
from sklearn.metrics import accuracy_score, precision_score, recall_score
def calculate_metrics(ytr, yte, yprtr, yprte):
    acc_train = accuracy_score(y_true=ytr, y_pred=yprtr)
    acc_test = accuracy_score (y_true=yte, y_pred=yprte)

    p = precision_score (y_true=yte, y_pred=yprte)
    r = recall_score(y_true=yte, y_pred=yprte)
    print(f"acc_train: {acc_train}, acc_test: {acc_test}, precision: {p}, recall: {r}")

    return acc_train, acc_test, p, r

# 1 . Naive bayse
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(xtr, ytr)

yprtr = gnb.predict(xtr)
yprte = gnb.predict(xte)
acc_train_gnb, acc_test_gnb, p_gnb, r_gnb = calculate_metrics(ytr, yte, yprtr, yprte)

# 2. KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=8, algorithm='kd_tree', leaf_size=28)
knn.fit(xtr, ytr)

yprtr = knn.predict(xtr)
yprte = knn.predict(xte)
acc_train_knn, acc_test_knn, p_knn, r_knn = calculate_metrics(ytr, yte, yprtr, yprte)
# 3. Decision tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=64, min_samples_split=2, criterion='entropy')
dt.fit(xtr, ytr)

yprtr = dt.predict(xtr)
yprte = dt.predict(xte)
acc_train_dt, acc_test_dt, p_dt, r_dt = calculate_metrics(ytr, yte, yprtr, yprte)


# 4.Random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=1000, max_depth=64, min_samples_split=8)
rf.fit(xtr, ytr)

yprtr = rf.predict(xtr)
yprte = rf.predict(xte)
acc_train_rf, acc_test_rf, p_rf, r_rf = calculate_metrics(ytr, yte, yprtr, yprte)

# 5.SVM
from sklearn.svm import SVC
svm = SVC(kernel='poly')
svm.fit(xtr, ytr)

yprtr = svm.predict(xtr)
yprte = svm.predict(xte)
acc_train_svm, acc_test_svm, p_svm, r_svm = calculate_metrics(ytr, yte, yprtr, yprte)

# 6. logistic regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(xtr, ytr)

yprtr = lr.predict(xtr)
yprte = lr.predict(xte)
acc_train_lr, acc_test_lr, p_lr, r_lr = calculate_metrics(ytr, yte, yprtr, yprte)

# 7. ANN
from sklearn.neural_network import MLPClassifier
ann = MLPClassifier (hidden_layer_sizes=256, activation='relu', solver='adam', batch_size=32)
ann.fit(xtr, ytr)

yprtr = ann.predict(xtr)
yprte = ann.predict(xte)
acc_train_ann, acc_test_ann, p_ann, r_ann = calculate_metrics(ytr, yte, yprtr, yprte)

### comparison
import matplotlib.pyplot as plt
acc_train = [acc_train_gnb, acc_train_knn, acc_train_dt, acc_train_rf, acc_train_svm, acc_train_lr, acc_train_ann]
title = ["GNB", "KNN", "DT", "RF", "SVM", "LR", "ANN" ]
colors = ["black", "red", "yellow", "orange", "green", "blue", "pink"]
plt.bar(title, acc_train, color = colors)
plt.grid()
plt.show()

acc_test = [acc_test_gnb, acc_test_knn, acc_test_dt, acc_test_rf, acc_test_svm, acc_test_lr, acc_test_ann]
title = ["GNB", "KNN", "DT", "RF", "SVM", "LR", "ANN" ]
colors = ["black", "red", "yellow", "orange", "green", "blue", "pink"]
plt.bar(title, acc_test, color = colors)
plt.grid()
plt.show()

p = [p_gnb, p_knn, p_dt, p_rf, p_svm, p_lr, p_ann]
title = ["GNB", "KNN", "DT", "RF", "SVM", "LR", "ANN" ]
colors = ["black", "red", "yellow", "orange", "green", "blue", "pink"]
plt.bar(title, p, color = colors)
plt.grid()
plt.show()

r = [r_gnb, r_knn, r_dt, r_rf, r_svm, r_lr, r_ann]
title = ["GNB", "KNN", "DT", "RF", "SVM", "LR", "ANN" ]
colors = ["black", "red", "yellow", "orange", "green", "blue", "pink"]
plt.bar(title, r, color = colors)
plt.grid()
plt.show()
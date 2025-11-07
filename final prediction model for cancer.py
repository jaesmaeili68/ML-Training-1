# this model is just trained and finally is save in a specific path in your system. 
# those text are in persian, you can translate them if you need
import pandas as pd
from sklearn.datasets import load_breast_cancer
bc = load_breast_cancer()

# Convert to data set
df = pd.DataFrame(bc.data, columns=bc.feature_names)
df['target'] = bc.target

print(df.head())
print(df.info())
print(df.describe())
##### این دستورات کمکت می‌کنن بفهمی:
# چند تا ویژگی داریم (۳۰ تا)
# چه نوع داده‌هایی هستن (float، int، ...)
# توزیع آماری هر ویژگی (میانگین، حداقل، حداکثر)
# مقدارهای گمشده وجود داره یا نه (در این دیتاست نداره)

# Target evaluation
print(bc.target_names)
print(df['target'].value_counts())



#plot class distribution
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='target', data=df)
plt.xticks([0,1], ['Malignant', 'Benign'])
plt.title('breast cancer label distribution')
plt.show()
# 🔹 با این نمودار می‌بینی که داده‌ها متعادل نیستند (خوش‌خیم‌ها کمی بیشترند).
# این موضوع روی دقت مدل اثر می‌گذاره (precision و recall مهم‌تر از accuracy می‌شن).
#--------------------------------------
# correlation matrix plot
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title('correlation matrix')
plt.show()
# 🔹 این نمودار نشون می‌ده کدوم ویژگی‌ها به هم وابسته‌ان.
# مثلاً “mean radius” با “mean perimeter” بسیار همبستگی داره.
#--------------------------------------

# pair plot (also known as a scatterplot matrix)
#This visualization is excellent for exploring relationships between multiple variables in a dataset.
sns.pairplot(df[['mean radius', 'mean texture', 'mean smoothness', 'target']], hue='target')
plt.show()
#🔹 با این نمودار می‌فهمی کدوم ویژگی‌ها بیشتر در جدا کردن خوش‌خیم/بدخیم مفیدن.


#بررسی توزیع ویژگی‌ها
df.hist(bins=20, figsize=(20,20))
plt.title('distribution of breast cancer features')
plt.show()
#🔹 کمک می‌کنه بفهمی داده‌ها نرمال هستن یا نه.

#=====================================================================
###EDA یعنی آماده‌سازی داده‌ها برای مدل‌سازی (Data Preprocessing).###

## preprocessing
from sklearn.model_selection import train_test_split
x = df.drop('target', axis=1)
y = df['target']



## Normalize
# چرا؟
# چون بعضی ویژگی‌ها مقدار خیلی بزرگ‌تری از بقیه دارن.
# مثلاً "mean area" ممکنه تا هزار باشه ولی "mean smoothness" تا ۰.۱.
# مدل‌هایی مثل Naive Bayes و KNN تحت تأثیر مقیاس قرار می‌گیرن.

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
x_scaled = scaler.fit_transform(x)

# حتماً fit فقط روی داده‌های train انجام می‌شه، و بعد همون scaler برای test استفاده می‌شه.

## Cross-Validation
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
import numpy as np
gnb = GaussianNB()
# اعتبارسنجی 5 بخشی (5-Fold Cross Validation)
scores = cross_val_score(gnb, x_scaled, y, cv=5, scoring='accuracy')
print("Accuracy for each fold:", scores)
print("Mean accuracy:", np.mean(scores))
print("Standard deviation:", np.std(scores))

## Train/test
xtr, xte, ytr, yte = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
gnb.fit(xtr, ytr)
y_pred = gnb.predict(xte)

print(f"xtr shape: {xtr.shape}, xte shape: {xte.shape}")
print(f"ytr shape: {ytr.shape}, yte shape: {yte.shape}")
#این بهت اطمینان می‌ده که داده‌هات درسته و آماده‌ی یادگیریه.

# accuracy
from sklearn.metrics import accuracy_score
print("Final test accuracy:", accuracy_score(yte, y_pred))

# Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(yte, y_pred)
print(f"Confusion matrics is : {cm}")
#[[TN FP]
# [FN TP]]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=bc.target_names)
disp.plot(cmap='Blues')
plt.show()

## ROC Curve و AUC  توان مدل در جدا کردن کلاس‌ها.
from sklearn.metrics import roc_curve, auc
# احتمال پیش‌بینی کلاس "سرطان بدخیم"
y_proba = gnb.predict_proba(xte)[:, 1]
# محاسبه‌ی نقاط منحنی
fpr, tpr, thresholds = roc_curve(yte, y_proba)
roc_auc = auc(fpr, tpr)
print(f"ROC_AUC is: {roc_auc}")

# رسم منحنی
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Naive Bayes')
plt.legend(loc="lower right")
plt.show()

##ذخیره مدل بعد از آموزش
import pickle
import os

save_path = r"E:\\Python Tutorial\\Teacher (ML)"
with open(os.path.join(save_path, 'breast_model.pkl'), 'wb') as file:
    pickle.dump(gnb, file)

with open(os.path.join(save_path, 'scaler.pkl'), 'wb') as file:
    pickle.dump(scaler, file)

print("Saved in:", save_path)
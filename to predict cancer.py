#in this code, the main trained model is saved somewhere else to avoid repeating and seeing all statistical graphs.
#the main trained model was named "final prediction model for cancer"
import pandas as pd
import pickle 
import os
from sklearn.datasets import load_breast_cancer
bc = load_breast_cancer()

# مسیر کامل پوشه‌ای که مدل رو ذخیره کردی
model_path = r"E:\Python Tutorial\Teacher (ML)"

# لود مدل و اسکیلر
with open(os.path.join(model_path, 'breast_model.pkl'), 'rb') as file:
    model = pickle.load(file)

with open(os.path.join(model_path, 'scaler.pkl'), 'rb') as file:
    scaler = pickle.load(file)

# خواندن فایل اکسل بیمار جدید
file_path = r"E:\\Python Tutorial\\Teacher (ML)\\new_patient.xlsx"
new_data = pd.read_excel(file_path)

print("New data shape:", new_data.shape)
print(new_data.head())

# تبدیل داده جدید با همان مقیاس آموزش
new_data_scaled = scaler.transform(new_data)

# پیش‌بینی کلاس
y_pred = model.predict(new_data_scaled)

# پیش‌بینی احتمال هر کلاس
y_prob = model.predict_proba(new_data_scaled)

print("Predicted class:", y_pred)
print("Prediction probabilities:", y_prob)

target_names = bc.target_names  # ['malignant', 'benign']

print("Predicted diagnosis:", target_names[y_pred[0]])
print("Probabilities -> Malignant: %.3f, Benign: %.3f" % (y_prob[0][0], y_prob[0][1]))

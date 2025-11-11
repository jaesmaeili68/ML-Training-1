# this code reads the main model file and then does prediction for a new patient
import pandas as pd
import pickle
import os
from sklearn.datasets import load_diabetes
db = load_diabetes()

#the main path for the model in your PC
model_path = r"E:\Python Tutorial\Teacher (ML)"

# load the model and scaler
with open(os.path.join(model_path, 'diabet_model.pkl'), 'rb') as file:
    model = pickle.load(file)
with open(os.path.join(model_path, 'scaler_diabet.pkl'), 'rb') as file:
    scaler = pickle.load(file)

#read new patient data
file_path = r'E:\\Python Tutorial\\Teacher (ML)\\new_diabetic_patient.xlsx'
new_data = pd.read_excel(file_path)

#shape of data
print ("new data shape:", new_data.shape)
print(new_data.head())

#data conversion
new_data_scaled = scaler.transform(new_data)

#prediction
y_pred = model.predict(new_data_scaled)

print("Predicted class:", y_pred)



    
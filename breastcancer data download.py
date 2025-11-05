import pandas as pd
from sklearn.datasets import load_breast_cancer
import os

# Load dataset
bc = load_breast_cancer()
df = pd.DataFrame(bc.data, columns=bc.feature_names)
df['target'] = bc.target

# Specify folder path
folder_path = r"E:\\Python Tutorial\\Teacher (ML)"  # use raw string (r"") for Windows paths
if not os.path.exists(folder_path):
    os.makedirs(folder_path)  # create folder if it doesn't exist

# Save Excel file in that folder
file_path = os.path.join(folder_path, "breast_cancer_dataset.xlsx")
df.to_excel(file_path, index=False)

print("Dataset saved at:", file_path)
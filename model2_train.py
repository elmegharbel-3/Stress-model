import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load the trained model and scaler
model = joblib.load("model_v2/stress_model.pkl")
scaler = joblib.load("model_v2/scaler.pkl")
train_folder = "cleaned"
# Load new dataset
train_files = os.listdir(train_folder)
for file in train_files:
    file_path = os.path.join(train_folder,file)
    print(file_path)
    # Change this to the new file path
    df_new = pd.read_csv(file_path)

    # Select features and target
    X_new = df_new[["hr", "spo2", "GSR(kohm)"]]
    y_new = df_new["stress_state"]

    # Standardize the new data using the existing scaler
    X_new_scaled = scaler.transform(X_new)  # Use the previously fitted scaler
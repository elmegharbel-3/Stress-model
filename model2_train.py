from sklearn.preprocessing import StandardScaler
import joblib
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the trained model and scaler
model = joblib.load("model_v3/stress_model_3.pkl")
scaler = joblib.load("model_v3/scaler_3.pkl")
train_folder = "cleaned"
# Load new dataset
skip_files = [r"cleaned\cleaned_30.csv",r"cleaned\cleaned_31.csv",r"cleaned\cleaned_32.csv",r"cleaned\cleaned_29.csv",r"cleaned\cleaned_28.csv"]
train_files = os.listdir(train_folder)
for file in train_files:
    file_path = os.path.join(train_folder,file)
    print(file_path)
    if file_path in skip_files:
        print("file skipped")
        continue
    
    df = pd.read_csv(file_path)
    # Select features and target
    X = df[["hr", "spo2", "GSR(kohm)"]]
    y = df["stress_state"]

    # Split data into training (80%) and testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y)
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    joblib.dump(model,"model_v3/stress_model_3.pkl")
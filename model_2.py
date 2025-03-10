import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib  # For saving the model

# Load dataset
file_path = "cleaned/cleaned_1.csv"  # Change this to your actual file path
df = pd.read_csv(file_path)

# Select features and target
X = df[["hr", "spo2", "GSR(kohm)"]]
y = df["stress_state"]

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the feature values (important for ML models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the model and scaler for later use
joblib.dump(model, "mstress_model_3.pkl")  # Save the trained model
joblib.dump(scaler, "scaler_3.pkl")  # Save the scaler (important for preprocessing new data)

print("Model and scaler saved successfully!")

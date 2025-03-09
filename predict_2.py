import joblib
import numpy as np

# Load the model and scaler
model = joblib.load("model_v2/stress_model.pkl")
scaler = joblib.load("model_v2/scaler.pkl")

# Example: Predict stress for a new sample
new_data = np.array([[100, 98, 100]])  # Example input: [HR, SpO2, GSR(kohm)]
new_data_scaled = scaler.transform(new_data)  # Apply the same scaling
prediction = model.predict(new_data_scaled)

# Print result
print("Predicted Stress State:", prediction[0])  # Output: 0 (no stress) or 1 (stress)
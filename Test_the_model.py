import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
# load the model and scaler put your path
model = joblib.load("model_v3/stress_model_3.pkl")   
scaler = joblib.load("model_v3/scaler_3.pkl")   
# load the file you will be testing with put your path
test_file_path = "cleaned/cleaned_32.csv"   
df_test = pd.read_csv(test_file_path)
# x => input data
X_new = df_test[["hr", "spo2", "GSR(kohm)"]]  
# y => stress state (0,1)
y_actual = df_test["stress_state"]  
# scale the x data with the scaler
X_new_scaled = scaler.transform(X_new)  
# this the model prediction based on x_new
y_pred = model.predict(X_new_scaled)
# comparison between the prediction and the acutal data
accuracy = accuracy_score(y_actual, y_pred)   
# display the success percentage
print(f"Model Success Percentage: {accuracy * 100:.2f}%")

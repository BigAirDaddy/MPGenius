import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import joblib

# Load model & scalers
model = keras.models.load_model('Model/automodel.keras')
scaler_X = joblib.load("Model/scaler_X.pkl")
scaler_y = joblib.load("Model/scaler_y.pkl")

# Define column names to match training data
column_names = ['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']

# 🚗 Test vehicles: [Cylinders, Displacement, Horsepower, Weight, Acceleration, Model Year, Origin]
# 🔹 Add real-world MPG for accuracy calculation
test_vehicles = [
    {"name": "1971 Chevy Nova 350", "features": [8, 350, 270, 2910, 8.4, 1971, 1], "actual_mpg": 15},
    {"name": "1996 Chevrolet Impala SS", "features": [8, 350, 260, 4200, 7.1, 1996, 1], "actual_mpg": 17},
    {"name": "2008 Honda Civic Si", "features": [4, 122, 197, 2954, 7.2, 2008, 3], "actual_mpg": 31},
    {"name": "2015 Ford F-150 5.0L V8", "features": [8, 302, 385, 4500, 6.2, 2015, 1], "actual_mpg": 18},
    {"name": "2018 Ford Mustang GT (V8)", "features": [8, 307, 460, 3705, 4.3, 2018, 1], "actual_mpg": 19}
]

# Storage for accuracy calculations
predictions = []
errors = []

# Run the gauntlet! 🔥
print("\n🚗 Running MPG Predictions...\n")
for vehicle in test_vehicles:
    name = vehicle["name"]
    actual_mpg = vehicle["actual_mpg"]

    # Convert test vehicle input to a Pandas DataFrame with column names
    X_input = pd.DataFrame([vehicle["features"]], columns=column_names)  

    # Scale the input (now formatted correctly)
    X_scaled = scaler_X.transform(X_input)

    # Predict MPG
    predicted_scaled = model.predict(X_scaled)[0][0]
    predicted_mpg = scaler_y.inverse_transform(np.array([[predicted_scaled]]))[0][0]

    # Calculate error
    error = abs(predicted_mpg - actual_mpg)
    errors.append(error)
    predictions.append((name, predicted_mpg, actual_mpg, error))

    print(f"🔹 {name}: Predicted {predicted_mpg:.2f} MPG | Actual {actual_mpg} MPG | Error: {error:.2f}")

# Compute overall accuracy
mean_absolute_error = np.mean(errors)
print("\n📊 **Model Accuracy Report**")
print(f"✅ Mean Absolute Error (MAE): {mean_absolute_error:.2f} MPG")

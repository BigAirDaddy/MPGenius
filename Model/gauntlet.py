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
column_names = ['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin','FI']

# 🚗 Test vehicles: [Cylinders, Displacement, Horsepower, Weight, Acceleration, Model Year, Origin]
# 🔹 Add real-world MPG for accuracy calculation
test_vehicles = [
    {"name": "2022 Renault Megane RS 1.8L Turbo", "features": [4, 110, 300, 3200, 5.7, 2022, 2, 1], "expected_mpg": 26},
{"name": "2017 Peugeot 3008 1.6L Turbo", "features": [4, 97, 165, 3500, 8.2, 2017, 2, 1], "expected_mpg": 31},
{"name": "2018 Citroën C5 Aircross 1.6L Turbo", "features": [4, 97, 165, 3500, 8.2, 2018, 2, 1], "expected_mpg": 30},
{"name": "2019 Skoda Octavia vRS 2.0L Turbo", "features": [4, 122, 245, 3200, 6.1, 2019, 2, 1], "expected_mpg": 28},
{"name": "2020 SEAT Leon Cupra 2.0L Turbo", "features": [4, 122, 290, 3300, 5.9, 2020, 2, 1], "expected_mpg": 26},
{"name": "2016 Opel Astra 1.4L Turbo", "features": [4, 85, 150, 2900, 8.3, 2016, 2, 1], "expected_mpg": 34},
{"name": "2020 Toyota Mark X 2.5L NA", "features": [6, 153, 203, 3400, 7.8, 2020, 3, 0], "expected_mpg": 28},
{"name": "2018 Honda Accord Euro 2.4L NA", "features": [4, 146, 200, 3200, 7.5, 2018, 3, 0], "expected_mpg": 31},
{"name": "2016 Mazda Atenza 2.5L NA", "features": [4, 152, 189, 3100, 8.2, 2016, 3, 0], "expected_mpg": 30},
{"name": "2015 Nissan Teana 2.5L NA", "features": [4, 152, 182, 3300, 8.5, 2015, 3, 0], "expected_mpg": 29},
{"name": "2019 Toyota Fortuner 2.7L NA", "features": [4, 165, 164, 4100, 10.5, 2019, 3, 0], "expected_mpg": 24},
{"name": "2017 Subaru Levorg 2.0L NA", "features": [4, 122, 150, 3200, 8.8, 2017, 3, 0], "expected_mpg": 30},
{"name": "2018 Mitsubishi Pajero 3.5L NA", "features": [6, 213, 250, 4700, 9.2, 2018, 3, 0], "expected_mpg": 20},
{"name": "2021 Toyota Hilux 2.7L NA", "features": [4, 165, 164, 4200, 10.2, 2021, 3, 0], "expected_mpg": 23},
{"name": "2019 Suzuki Grand Vitara 2.4L NA", "features": [4, 146, 166, 3500, 9.6, 2019, 3, 0], "expected_mpg": 26},
{"name": "2015 Hyundai i40 2.0L NA", "features": [4, 122, 164, 3100, 9.2, 2015, 3, 0], "expected_mpg": 31},
{"name": "2013 Renault Duster 2.0L NA", "features": [4, 122, 138, 3150, 10.0, 2013, 2, 0], "expected_mpg": 29},
{"name": "2020 Peugeot 508 1.6L NA", "features": [4, 97, 163, 3300, 8.5, 2020, 2, 0], "expected_mpg": 34},
{"name": "2017 Citroën C4 1.6L NA", "features": [4, 97, 115, 2800, 9.7, 2017, 2, 0], "expected_mpg": 35},
{"name": "2018 Fiat Tipo 1.4L NA", "features": [4, 85, 95, 2600, 10.5, 2018, 2, 0], "expected_mpg": 38},
{"name": "2016 Opel Insignia 2.0L NA", "features": [4, 122, 160, 3400, 8.9, 2016, 2, 0], "expected_mpg": 31},
{"name": "2021 Skoda Superb 2.0L NA", "features": [4, 122, 188, 3450, 7.9, 2021, 2, 0], "expected_mpg": 30},
{"name": "2014 Lada Granta 1.6L NA", "features": [4, 97, 106, 2700, 10.8, 2014, 2, 0], "expected_mpg": 36}

]
# Storage for accuracy calculations
predictions = []
errors = []

# Run the gauntlet! 🔥
print("\n🚗 Running MPG Predictions...\n")
for vehicle in test_vehicles:
    name = vehicle["name"]
    actual_mpg = vehicle["expected_mpg"]

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

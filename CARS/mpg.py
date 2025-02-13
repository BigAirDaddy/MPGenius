import tensorflow as tf
import numpy as np 
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import joblib 

model= keras.models.load_model('Model/automodel.keras')

scaler_X = joblib.load("Model/scaler_X.pkl")
scaler_y = joblib.load("Model/scaler_y.pkl")

cylinders = input('Input the number of cylinders the vehicle has: ')
displacement = input('Input the displacement in cubic inches: ')
horsepower = input('Input the Horsepower the vehicle has: ')
weight = input('Input the weight of the vehicle: ')
acceleration = input('Input the 0-60 mph time: ')
model_year = input('Input the model year: ')
origin = input('Input the origin of the vehicle(1 for US 2 for Europe 3 for Asia): ')

sample_car = np.array([[cylinders,displacement,horsepower,weight,acceleration,model_year,origin]])
feature_names = ["Cylinders", "Displacement", "Horsepower", "Weight", "Acceleration", "Model Year", "Origin"]
sample_car_df = pd.DataFrame(sample_car, columns=feature_names)

sample_car_scaled = scaler_X.transform(sample_car_df)

nn_prediction_scaled = model.predict(sample_car_scaled)[0][0]

nn_prediction = scaler_y.inverse_transform(np.array([[nn_prediction_scaled]]))[0][0]

print(f"📡 Predicted MPG: {nn_prediction:.2f}")

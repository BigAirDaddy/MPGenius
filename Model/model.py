import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from tensorflow.keras.callbacks import EarlyStopping
import joblib


df = pd.read_csv('Data/auto.csv')

df = df.drop(columns=["Car Name"], errors="ignore")

df = df.apply(pd.to_numeric, errors="coerce")

df.dropna(inplace=True)


X = df.drop(columns=["MPG"]) # Inputs (car specs)
y = df["MPG"].values



scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu',input_shape=(X.shape[1],)),
    keras.layers.Dense(64,activation='relu'),
    keras.layers.Dense(1)
])

optimizer = keras.optimizers.Adam(learning_rate=0.00011)
model.compile(optimizer=optimizer, loss='mse')
early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

model.fit(X_train, y_train, epochs=150, batch_size=8, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Linear model to compare outputs with 
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
baseline_loss = np.mean((lr_model.predict(X_test) - y_test) ** 2)
print(f"🔹 Baseline Linear Regression Loss: {baseline_loss}")


sample_car = np.array([[8, 350, 275, 5600, 6.5, 1996, 1]])  # Example input to verify accuracy
print("Original Input Data:", sample_car)

sample_car = pd.DataFrame(sample_car, columns=X.columns)  
sample_car_scaled = scaler.transform(sample_car)
print("Scaled Input Data:", sample_car_scaled)



nn_prediction_scaled = model.predict(sample_car_scaled)[0][0]
nn_prediction = scaler_y.inverse_transform(nn_prediction_scaled.reshape(-1, 1))[0][0]




lr_prediction_scaled = lr_model.predict(sample_car_scaled)[0]
lr_prediction = scaler_y.inverse_transform(lr_prediction_scaled.reshape(-1, 1))[0][0]


print(f"📡 Neural Network Prediction: {nn_prediction:.2f} MPG")
print(f"✅ Linear Regression Prediction: {lr_prediction:.2f} MPG")


joblib.dump(scaler, "Model/scaler_X.pkl")  
joblib.dump(scaler_y, "Model/scaler_y.pkl")

model.save("Model/automodel.keras")
print('Model was succesfully saved!')

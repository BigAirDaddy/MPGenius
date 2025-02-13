# 🚗 MPGenius – AI-Powered Fuel Efficiency Predictor  

MPGenius is a machine learning model that predicts a vehicle's **miles per gallon (MPG)** based on its **engine specs, weight, acceleration, model year, and origin**. Built with **TensorFlow and scikit-learn**, this project leverages **neural networks and regression models** to provide accurate fuel efficiency estimates.  

---

##  Features  
-  **Predicts fuel efficiency (MPG)** based on key vehicle attributes  
-  **Uses Neural Networks & Linear Regression** for comparison  
- **Trained on real-world vehicle data** (classic & modern cars)  
-  **Supports new vehicle predictions** within realistic data ranges  
-  **Fully open-source**, built with **Python, TensorFlow, and scikit-learn**  

---

##  How It Works  
1. **Input vehicle specifications**  
2. **The trained AI model processes and normalizes the data**  
3. **Predicts fuel efficiency (MPG) using neural networks**  
4. **Compares results with a traditional linear regression model**  

---

## 🛠 Tech Stack  
- **Python** (Pandas, NumPy, scikit-learn)  
- **TensorFlow / Keras** (Neural Network)  
- **StandardScaler** (Data normalization)  
- **Joblib** (Saving & loading trained models)  

---

##  Installation & Usage  

### 1. Clone the Repository  
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/MPGenius.git
cd MPGenius
## 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## 3. Train the Model (Optional – Pretrained Model Available)
```bash
python train.py
```

## 4. Run the Predictor
```bash
python predict.py
```

## 📥 Example Input
```less
Input the number of cylinders the vehicle has: 6  
Input the displacement (cu in): 250  
Input the Horsepower: 145  
Input the weight (lbs): 2825  
Input the 0-60 acceleration time (sec): 12.7  
Input the model year: 1971  
Input the origin (1 for US, 2 for Europe, 3 for Asia): 1  
```

## 📊 Example Output
```objectivec
📡 Predicted MPG: 18.57
✅ Linear Regression Prediction: 19.2 MPG
```

## 📈 Sample Predictions
| Car | Predicted MPG | Actual MPG |
|-------------------------------|---------------|-------------|
| 1971 Chevy Nova 350 (V8) | 14.9 MPG | ~15 MPG |
| 1996 Chevrolet Impala SS (V8) | 17.6 MPG | ~17 MPG |
| 2008 Honda Civic Si (I4) | 38.2 MPG | ~31 MPG |

🚀 More training data improves accuracy over time!

---



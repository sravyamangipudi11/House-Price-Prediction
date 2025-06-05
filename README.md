# 🏠 House Price Prediction - Streamlit App

A simple and interactive web application to predict house prices using machine learning. Built with **Python**, **Streamlit**, and **scikit-learn**.


---

## 🚀 Features

- Predict house prices based on user inputs
- Clean and simple UI with Streamlit
- Trained on real housing data
- Fully open-source and ready to deploy

---

## 📂 Project Structure

.
├── app.py # Streamlit app
├── house_price_prediction.ipynb # Jupyter notebook (EDA + training)
├── house_data.csv # Dataset
├── house_price_model.pkl # Trained model
├── scaler.pkl # Scaler object for preprocessing
├── full_features.pkl # Feature columns used during training
├── model_metadata.json # Model metadata
└── README.md # Project overview

---

## 🧠 Machine Learning Model

- Model: `RandomForestRegressor`
- Preprocessing: StandardScaler
- Evaluation Metric: RMSE / R²
- Dataset: Cleaned housing data

---

## ▶️ How to Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/sravyamangipudi11/House-Price-Prediction.git
   cd House-Price-Prediction
2. Create a virtual environment and install dependencies:
    ```bash
    python -m venv venv
    venv\Scripts\activate      
## On Windows 
    pip install -r requirements.txt

## 🛠️Tools Used
Python

Streamlit

scikit-learn

pandas, numpy, matplotlib, seaborn

## 🙋‍♀️Author
Sravya Mangipudi

## License
This project is open-source under the MIT License.

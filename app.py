import numpy as np
import pandas as pd
import streamlit as st
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap

custom_css = """
<style>
/* Add margin below all subheaders for spacing */
h2, h3, h4 {
    margin-bottom: 1rem;
}

/* Add spacing around main sections */
section.main-section {
    margin-bottom: 2rem;
    padding: 1rem;
}

/* Style buttons - bigger padding, smooth hover */
button[class*="stButton"] > button {
    padding: 12px 24px !important;
    font-size: 1.1rem !important;
    border-radius: 12px !important;
    transition: background-color 0.3s ease, transform 0.2s ease;
}
button[class*="stButton"] > button:hover {
    background-color: #e67300 !important;
    transform: scale(1.05);
}

/* Inputs & selects padding */
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] select,
div[data-testid="stTextArea"] textarea {
    padding: 10px !important;
    font-size: 1rem !important;
}

/* Add margin bottom for inputs and selects */
div[data-testid="stTextInput"],
div[data-testid="stNumberInput"],
div[data-testid="stSelectbox"],
div[data-testid="stTextArea"] {
    margin-bottom: 1rem;
}
</style>

<style>
/* Brown sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(90deg, #240046, #3c096c);
    color: #fb5607;
}

/* Cream/beige background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #d1be9c, #184e77);
    color: #580c1f;
}

/* Headers indigo */
h1, h2, h3, h4, h5, h6 {
    color: indigo;
}

/* Buttons orange rounded */
button[class*="stButton"] > button {
    background-color: #ff7f00;
    border-radius: 15px;
    color: white;
    font-weight: bold;
    border: none;
    padding: 8px 20px;
    transition: background-color 0.3s ease;
}
button[class*="stButton"] > button:hover {
    background-color: #e67300;
}

/* Custom colors for different sliders by aria-label */
div[data-testid="stSlider"][aria-label="Overall Qual"] .rc-slider-track {
    background-color: #1f77b4 !important;  /* blue */
}
div[data-testid="stSlider"][aria-label="Overall Qual"] .rc-slider-handle {
    border-color: #1f77b4 !important;
}

div[data-testid="stSlider"][aria-label="Year Built"] .rc-slider-track {
    background-color: #ff7f0e !important; /* orange */
}
div[data-testid="stSlider"][aria-label="Year Built"] .rc-slider-handle {
    border-color: #ff7f0e !important;
}

div[data-testid="stSlider"][aria-label="Garage Cars"] .rc-slider-track {
    background-color: #2ca02c !important; /* green */
}
div[data-testid="stSlider"][aria-label="Garage Cars"] .rc-slider-handle {
    border-color: #2ca02c !important;
}

/* Number inputs background & border */
div[data-testid="stNumberInput"][aria-label="Gr Liv Area (sq ft)"] input {
    border: 2px solid #d2691e; /* chocolate brown */
    border-radius: 10px;
    color: black;
    background-color: #fff8dc; /* cornsilk */
    font-weight: 600;
}

div[data-testid="stNumberInput"][aria-label="Lot Area (sq ft)"] input {
    border: 2px solid #8b4513; /* saddle brown */
    border-radius: 10px;
    color: black;
    background-color: #f5f5dc; /* beige */
    font-weight: 600;
}

/* Selectbox style */
div[data-testid="stSelectbox"][aria-label="Neighborhood"] select {
    border: 2px solid #a0522d; /* sienna */
    border-radius: 10px;
    color: black;
    background-color: #fdf5e6; /* old lace */
    font-weight: 600;
    padding: 4px 8px;
}

/* Text input boxes */
div[data-testid="stTextInput"] input {
    background-color: #f1faeeff;  /* honeydew */
    color: #1d3557ff;             /* berkeley-blue */
    border: 3px solid #e63946ff; /* red-pantone - thicker border */
    border-radius: 8px;
    padding: 10px;
}

/* Multi-line text areas */
div[data-testid="stTextArea"] textarea {
    background-color: #f1faeeff;
    color: #1d3557ff;
    border: 3px solid #e63946ff;
    border-radius: 8px;
    padding: 10px;
}

/* Number inputs */
div[data-testid="stNumberInput"] input {
    background-color: #a8dadcff;
    color: #1d3557ff;
    border: 3px solid #e63946ff;
    border-radius: 8px;
    padding: 10px;
}

/* Selectbox dropdown */
div[data-testid="stSelectbox"] select {
    background-color: #f1faeeff;
    color: #1d3557ff;
    border: 3px solid #e63946ff;
    border-radius: 8px;
    padding: 10px;
}

/* On focus: highlight border */
div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus,
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stSelectbox"] select:focus {
    border-color: #a8dadcff !important; /* non-photo-blue */
    outline: none;
    box-shadow: 0 0 5px 1px #a8dadcff;
}

</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)




# Load model and scaler
model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load full feature list
if os.path.exists('full_features.pkl'):
    full_features = joblib.load('full_features.pkl')
else:
    st.error("Missing 'full_features.pkl'. Please upload or generate it.")
    st.stop()

st.title("🏠 House Price Prediction")
st.sidebar.header("Specify House Features")

# --- Inputs ---
overall_qual = st.sidebar.slider("Overall Qual", 1, 10, 5)
gr_liv_area = st.sidebar.number_input("Gr Liv Area (sq ft)", 300, 4000, 1500)
year_built = st.sidebar.slider("Year Built", 1870, 2022, 2000)
garage_cars = st.sidebar.slider("Garage Cars", 0, 5, 2)
lot_area = st.sidebar.number_input("Lot Area (sq ft)", 1000, 200000, 10000)

neighborhood_options = ['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst']
neighborhood = st.sidebar.selectbox("Neighborhood", neighborhood_options)

# Initialize input DataFrame with zeros for all features
input_df = pd.DataFrame(np.zeros((1, len(full_features))), columns=full_features)

# Assign numerical inputs - make sure keys match full_features exactly
input_df.at[0, 'Overall Qual'] = overall_qual
input_df.at[0, 'Gr Liv Area'] = gr_liv_area
input_df.at[0, 'Year Built'] = year_built
input_df.at[0, 'Garage Cars'] = garage_cars
input_df.at[0, 'Lot Area'] = lot_area

# One-hot encode neighborhood
neigh_col = f"Neighborhood_{neighborhood}"
if neigh_col in input_df.columns:
    input_df.at[0, neigh_col] = 1

# Show input preview
st.write("### Input Features Preview")
st.dataframe(input_df)

# Scale inputs for prediction
scaled_input = scaler.transform(input_df)

# Predict house price
price_pred = model.predict(scaled_input)

st.subheader("Predicted House Price")
st.write(f"💰 ${price_pred[0]:,.2f}")

# Add prediction to history on button click to avoid duplicate appends on reruns
if 'pred_history' not in st.session_state:
    st.session_state.pred_history = []

if st.button("Add to Prediction History"):
    current_record = input_df.copy()
    current_record['Predicted Price'] = price_pred[0]
    st.session_state.pred_history.append(current_record)

# Show history if exists
if st.session_state.pred_history:
    history_df = pd.concat(st.session_state.pred_history).reset_index(drop=True)
    st.subheader("Prediction History")
    st.dataframe(history_df)

# Model performance metrics (hardcoded)
st.subheader("Model Performance Metrics (Test Set)")
mae = 15828.29
rmse = 26440.21
r2 = 0.913

st.write(f"Mean Absolute Error (MAE): ${mae:,.2f}")
st.write(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
st.write(f"R² Score: {r2:.3f}")

# Feature Importance plot
st.subheader("Feature Importance")
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    feature_names = full_features

    feat_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False).head(10)

    fig, ax = plt.subplots()
    sns.barplot(x='importance', y='feature', data=feat_imp_df, ax=ax)
    ax.set_title("Top 10 Feature Importances")
    st.pyplot(fig)
else:
    st.write("Feature importance not available for this model type.")

# SHAP Explanation
st.subheader("SHAP Explanation Plot")

try:
    explainer = shap.TreeExplainer(model)

    # Use original (unscaled) input for SHAP explanation
    shap_values = explainer.shap_values(input_df)

    fig_shap = plt.figure()
    shap.summary_plot(shap_values, input_df, feature_names=full_features, plot_type="bar", show=False)
    st.pyplot(fig_shap)

except Exception as e:
    st.write("SHAP explanation could not be generated:", e)

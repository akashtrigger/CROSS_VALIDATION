import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Medical Disease Prediction", layout="centered")
st.title("🩺 Medical Disease Prediction App")

# -------------------------
# CREATE DATASET (NO CSV NEEDED)
# -------------------------
st.subheader("📊 Medical Dataset (Auto Generated)")

np.random.seed(42)
df = pd.DataFrame({
    "Age": np.random.randint(20, 80, 200),
    "BP": np.random.randint(80, 180, 200),
    "Cholesterol": np.random.randint(150, 300, 200),
    "HeartRate": np.random.randint(60, 120, 200),
    "Disease": np.random.randint(0, 2, 200)
})

st.dataframe(df.head())

# -------------------------
# SPLIT DATA
# -------------------------
X = df.drop("Disease", axis=1)
y = df["Disease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# SCALING
# -------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# MODEL TRAINING
# -------------------------
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# -------------------------
# CROSS VALIDATION
# -------------------------
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

st.subheader("📈 Model Performance")
st.write("Cross Validation Scores:", cv_scores)
st.write("Average CV Accuracy:", round(cv_scores.mean(), 2))

# -------------------------
# TEST ACCURACY
# -------------------------
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
st.success(f"Test Accuracy: {round(accuracy, 2)}")

# -------------------------
# SAVE MODEL & SCALER
# -------------------------
joblib.dump(model, "medical_disease_model.pkl")
joblib.dump(scaler, "scaler.pkl")
st.info("💾 Model & Scaler saved successfully")

# -------------------------
# USER INPUT FOR PREDICTION
# -------------------------
st.subheader("🔍 Predict Disease")

age = st.slider("Age", 20, 80, 40)
bp = st.slider("Blood Pressure", 80, 180, 120)
chol = st.slider("Cholesterol", 150, 300, 200)
hr = st.slider("Heart Rate", 60, 120, 80)

input_data = np.array([[age, bp, chol, hr]])
input_scaled = scaler.transform(input_data)

if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.error("⚠ Disease Detected")
    else:
        st.success("✅ No Disease Detected")

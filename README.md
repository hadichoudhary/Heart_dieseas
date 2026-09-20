# ❤️ CardioPulse — Heart Disease Prediction Web App

A responsive, clinical-grade Heart Disease Risk Prediction web application built with **Streamlit** and **Scikit-Learn (K-Nearest Neighbors Classifier)**.

---

## 📌 Project Overview

This application predicts the likelihood of heart disease in patients based on 11 diagnostic and cardiovascular markers. The predictive engine utilizes a pre-trained **K-Nearest Neighbors (KNN)** model coupled with a dynamic `StandardScaler` pipeline.

### ✨ Key Features
- **Modern & Device-Friendly UI**: Fully responsive layout tailored for mobile phones, tablets, and desktop workstations.
- **Quick-Load Clinical Presets**: 1-click testing with pre-configured healthy and high-risk patient profiles.
- **Intelligent Preprocessing**: Automatically handles dummy encodings (`drop_first=True`) and two-stage standard scaling matching the model's training distribution.
- **Diagnostic Insights**: Probability assessment gauges, class breakdown charts, and clinical flag warnings.
- **Model Vector Inspector**: Expandable inspection tool to view exact 15D numerical vectors.

---

## 🗂️ Project Structure

```
model_dep/
├── app.py              # Streamlit web application & interface
├── KNN_heart.pkl       # Trained KNeighborsClassifier model
├── scaler.pkl          # Trained StandardScaler pipeline
├── columns.pkl         # Feature column alignment list (15 dimensions)
├── requirements.txt    # Python dependencies
├── .gitignore          # Git exclusion rules
└── README.md           # Project documentation
```

---

## 📊 Features & Model Pipeline

The model expects a 15-dimensional input vector derived from clinical features:

1. **Age**: Patient's age in years
2. **Sex**: Biological sex (`Male`, `Female`)
3. **ChestPainType**: `ATA` (Atypical Angina), `NAP` (Non-Anginal Pain), `ASY` (Asymptomatic), `TA` (Typical Angina)
4. **RestingBP**: Resting blood pressure (mm Hg)
5. **Cholesterol**: Serum cholesterol (mg/dl)
6. **FastingBS**: Fasting blood sugar > 120 mg/dl (0 or 1)
7. **RestingECG**: `Normal`, `ST` (ST-T abnormality), `LVH` (Left Ventricular Hypertrophy)
8. **MaxHR**: Maximum heart rate achieved during exercise
9. **ExerciseAngina**: Exercise-induced angina (`Yes`, `No`)
10. **Oldpeak**: ST depression induced by exercise relative to rest
11. **ST_Slope**: Slope of the peak exercise ST segment (`Up`, `Flat`, `Down`)

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/hadichoudhary/Heart_dieseas.git
cd Heart_dieseas
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application
```bash
streamlit run app.py
```

The application will be accessible at `http://localhost:8501`.

---

## 🔒 Medical Disclaimer
*This system is intended for exploratory, educational, and clinical research decision-support purposes only and does not substitute professional medical diagnosis.*

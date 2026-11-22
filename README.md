# CardioPulse AI — Smart Heart Health Prediction System

> **Slogan:** *Predict. Understand. Stay Aware.*  
> **Tagline:** *Smart Heart Health Prediction System*  
> **Live Demo:** `[Live Demo Placeholder: https://cardiopulse-ai.streamlit.app]`

A production-grade, clinical-grade machine learning application for predicting the presence or absence of coronary heart disease based on patient physiological and diagnostic indicators. Built with Scikit-Learn, Joblib, and Streamlit.

---

## 1. Project Introduction & Overview

Cardiovascular diseases (CVDs) remain the leading cause of morbidity and mortality globally. Early detection and proactive risk assessment significantly improve clinical outcomes by enabling timely interventions. **CardioPulse AI** implements an end-to-end Machine Learning pipeline and an interactive web application that evaluates 13 clinical risk factors—including resting blood pressure, serum cholesterol, electrocardiographic results, exercise-induced angina, and fluoroscopy vessel counts—to predict whether heart disease is **present** or **absent**.

---

## 2. Problem Statement

Given a patient's clinical diagnostic features, determine the probability and binary classification of heart disease risk. The objective is to produce a model that balances high sensitivity (recall) to minimize false negatives (critical in healthcare) with strong overall accuracy and calibration.

---

## 3. Objectives

- **Robust Preprocessing**: Handle missing numerical values via median imputation and normalize features with standard scaling.
- **Fair Benchmarking**: Evaluate multiple classification algorithms using Repeated Stratified 5-Fold Cross-Validation.
- **Leakage-Free Optimization**: Perform Nested Cross-Validation (outer 5-fold, inner 3-fold) for unbiased hyperparameter tuning.
- **Interpretability & Transparency**: Provide permutation feature importance and exact prediction probabilities.
- **Interactive Clinical Tool**: Deliver a user-friendly Streamlit web interface with real-time validation and educational guidance.

---

## 4. Dataset Information

- **Dataset**: Statlog (Heart) Dataset
- **Source**: OpenML (Dataset ID: `53`)
- **Total Records**: 270 patient instances
- **Target Variable**: Binary classification — `absent` (0) vs. `present` (1)
- **Features (13 clinical indicators)**:
  1. `age`: Age in years (Range: 29.0 - 77.0, Median: 55.0)
  2. `sex`: Biological sex (0 = Female, 1 = Male)
  3. `chest`: Chest pain type (1 = Typical angina, 2 = Atypical angina, 3 = Non-anginal pain, 4 = Asymptomatic)
  4. `resting_blood_pressure`: Resting blood pressure in mm Hg (Range: 94.0 - 200.0, Median: 130.0)
  5. `serum_cholestoral`: Serum cholesterol in mg/dl (Range: 126.0 - 564.0, Median: 245.0)
  6. `fasting_blood_sugar`: Fasting blood sugar > 120 mg/dl (0 = <= 120 mg/dl, 1 = > 120 mg/dl)
  7. `resting_electrocardiographic_results`: Resting ECG (0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy)
  8. `maximum_heart_rate_achieved`: Maximum heart rate achieved during exercise (Range: 71.0 - 202.0, Median: 153.5)
  9. `exercise_induced_angina`: Exercise-induced angina (0 = No, 1 = Yes)
  10. `oldpeak`: ST depression induced by exercise relative to rest (Range: 0.0 - 6.2, Median: 0.8)
  11. `slope`: Peak exercise ST segment slope (1 = Upsloping, 2 = Flat, 3 = Downsloping)
  12. `number_of_major_vessels`: Number of major vessels (0–3) colored by fluoroscopy (Range: 0.0 - 3.0, Median: 0.0)
  13. `thal`: Thalassemia category (3 = Normal, 6 = Fixed defect, 7 = Reversible defect)

---

## 5. Technologies & Stack

- **Core Language**: Python 3.10 - 3.13
- **Machine Learning**: Scikit-Learn `1.6.1`
- **Model Deserialization**: Joblib `1.5.3`
- **Web Application Framework**: Streamlit `1.67.0`
- **Data Manipulation**: Pandas, NumPy
- **Visual Analytics**: Matplotlib, Seaborn
- **Styling**: Theme-aware responsive CSS with CSS variables (`var(--text-color)`)

---

## 6. Machine Learning Algorithms Evaluated

Five model architectures were evaluated on identical stratified cross-validation folds:
1. **DummyClassifier (Baseline)**: Majority class baseline.
2. **Logistic Regression (Selected Best Model)**: Tuned L2 regularization ($C = 0.01$), class-weighted for sensitivity.
3. **Random Forest Classifier**: Ensemble of 300 decision trees with balanced class weights.
4. **Histogram-based Gradient Boosting Classifier**: Tree boosting algorithm.
5. **K-Nearest Neighbors (KNN)**: Distance-based neighbor classifier.

---

## 7. Methodology & 8-Stage Training Pipeline

1. **Stage 1 — Data Ingestion & EDA**: Ingested OpenML ID 53, deduplicated records, generated correlation matrix and balance charts.
2. **Stage 2 — Leak-Free Data Splitting**: Stratified 80/20 train/test split (`random_state=42`), reserving 54 unseen test samples for final evaluation.
3. **Stage 3 — Modular Preprocessing**: `ColumnTransformer` applying `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
4. **Stage 4 — Baseline Architecture Benchmarking**: Repeated Stratified 5-Fold CV with 3 repeats (15 evaluation folds).
5. **Stage 5 — Nested Cross-Validation & Tuning**: Outer 5-fold CV, inner 3-fold GridSearchCV over $C \in [0.01, 0.1, 1, 10, 100]$.
6. **Stage 6 — Pipeline Ablation Study**: Quantified the contribution of feature scaling, hyperparameter tuning, and class weighting.
7. **Stage 7 — Final Test Set Evaluation**: Single evaluation on 54 holdout test samples.
8. **Stage 8 — Interpretability & Model Persistence**: Permutation importance computation, exporting `model.joblib` and `meta.json`.

---

## 8. Actual Model Evaluation Results

### Baseline Model Comparison (5-Fold Repeated CV)
| Model | ROC-AUC | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Selected)** | **0.906 ± 0.052** | **0.822 ± 0.059** | **0.802 ± 0.085** | **0.812 ± 0.105** | **0.802 ± 0.067** |
| Random Forest | 0.901 ± 0.057 | 0.836 ± 0.068 | 0.847 ± 0.102 | 0.784 ± 0.114 | 0.808 ± 0.083 |
| HistGradientBoosting | 0.897 ± 0.061 | 0.798 ± 0.086 | 0.788 ± 0.120 | 0.770 ± 0.082 | 0.774 ± 0.086 |
| K-Nearest Neighbors | 0.895 ± 0.044 | 0.830 ± 0.056 | 0.839 ± 0.081 | 0.774 ± 0.104 | 0.800 ± 0.070 |
| Dummy Classifier | 0.500 ± 0.000 | 0.556 ± 0.005 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |

### Nested Cross-Validation (Hyperparameter Tuning)
- **Nested CV ROC-AUC**: `0.906 ± 0.038`
- **Optimal Hyperparameters**: `clf__C: 0.01` (with `penalty='l2'`, `class_weight='balanced'`)

### Final Hold-Out Test Set Performance (54 Unseen Samples)
- **ROC-AUC**: **90.97%** (`0.9097`)
- **Accuracy**: **83.33%** (`0.8333`)
- **Recall (Sensitivity)**: **87.50%** (`0.8750`)
- **Precision**: **77.78%** (`0.7778`)
- **F1-Score**: **82.35%** (`0.8235`)

### Ablation Study Results
| Pipeline Variant | ROC-AUC (mean ± std) |
| :--- | :---: |
| **Full Tuned Pipeline** | **0.913 ± 0.048** |
| Without Standard Scaling | 0.860 ± 0.045 |
| Default Hyperparameters | 0.906 ± 0.052 |
| Without Class Weighting | 0.914 ± 0.047 |

---

## 9. Project Structure

```
Heart_Disease_Project/
├── app.py                         # Streamlit interactive web application
├── model.joblib                   # Pre-trained Scikit-Learn pipeline
├── meta.json                      # Metadata, feature definitions, and valid ranges
├── requirements.txt               # Pinned package dependencies (scikit-learn==1.6.1)
├── sample_patients.csv            # Real OpenML 53 patient examples for quick comparison
├── README.md                      # Comprehensive project documentation
├── .gitignore                     # Git ignore rules
├── .streamlit/                    # Streamlit configuration
│   └── config.toml                # Dark theme and layout styling configuration
├── Heart_Disease_Notebook.ipynb   # Executed Jupyter/Colab pipeline notebook
├── ML_Pipeline_Colab.ipynb        # Original Colab pipeline notebook
├── assets/                        # Brand identity and cinematic photography assets
│   ├── realistic_heart.png        # Seamless 3D anatomical beating heart (from heart.avif)
│   ├── male_anatomy_3d.png        # 3D full-body male anatomical model (isolated transparent)
│   ├── female_anatomy_3d.png      # 3D full-body female anatomical model (isolated transparent)
│   ├── hero.jpg                   # Cardiac catheterization suite photo (<250 KB)
│   ├── stethoscope.jpg            # Diagnostic stethoscope photo (<250 KB)
│   ├── consultation.jpg           # Physician consultation photo (<250 KB)
│   ├── lifestyle.jpg              # Aerobic exercise & healthy lifestyle photo (<250 KB)
│   ├── blood_pressure.jpg         # Blood pressure examination photo (<250 KB)
│   ├── CREDITS.md                 # Attribution, photographer, source URLs & licenses
│   ├── cardiopulse_logo.svg       # Full horizontal brand lockup
│   ├── cardiopulse_icon.svg       # Anatomical heart + ECG + AI nodes mark
│   ├── favicon.png                # Browser tab icon
│   └── favicon.ico                # Windows icon format
├── figures/                       # Empirical visualization assets
│   ├── eda.png                    # Class distribution and feature correlation heatmap
│   ├── importance.png             # Permutation feature importance chart
│   └── test_eval.png              # Test confusion matrix and ROC curve
└── results/                       # Empirical results and metrics
    ├── baselines.csv              # Cross-validation comparison across algorithms
    ├── tuning.json                # Hyperparameter tuning results
    ├── test_metrics.json          # Final holdout test metrics
    ├── ablation.csv               # Pipeline ablation study metrics
    ├── eda_summary.csv            # Summary statistics of features
    └── errors.csv                 # Misclassified test instance analysis
```

---

## 10. Installation & Local Execution

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13

### Step 1: Open the Repository
```bash
cd Heart_Disease_Project
```

### Step 2: Create & Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Web Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 11. Streamlit Community Cloud Deployment

This project is structured for 1-click deployment on **Streamlit Community Cloud**:
1. Create a public repository on GitHub (e.g. `cardiopulse-ai`).
2. Push your files following logical commits:
   - `commit 1`: `Initial commit: project structure and dataset metadata`
   - `commit 2`: `Add trained model artifacts and results`
   - `commit 3`: `Add empirical evaluation figures and notebook`
   - `commit 4`: `Add CardioPulse AI brand assets and icons`
   - `commit 5`: `Implement modern clinical Streamlit web application in app.py`
   - `commit 6`: `Add pinned requirements.txt and .gitignore`
   - `commit 7`: `Update comprehensive documentation in README.md`
3. Visit [share.streamlit.io](https://share.streamlit.io), connect your GitHub account.
4. Select the repository, set branch to `main`, and **Main file path** to `app.py`.
5. Click **Deploy**. The app uses relative paths, contains no secrets, and executes automatically.

---

## 12. Limitations & Future Scope

### Limitations
- **Sample Cohort**: Trained on 270 instances from the Statlog Heart dataset; larger multi-center cohorts are needed for clinical generalization.
- **Data Modality**: Analyzes tabular numerical and categorical diagnostic markers; does not directly process medical imaging scans (DICOM) or continuous telemetry feeds.
- **Scope**: Designed as an engineering decision-support showcase and educational reference.

### Future Scope
- Integration with FHIR / HL7 clinical health record standards.
- Real-time wearable telemetry data ingestion (e.g., continuous smartwatch heart rate and pulse oximetry).
- Tree-explainer SHAP force plots for local per-patient risk attribution.

---

## 13. Disclaimer

> **Educational & Research Notice**: This application and machine learning model are developed strictly for educational, informational, and demonstration purposes. It does not provide medical diagnoses, treatment recommendations, or clinical advice. Always consult a licensed medical professional or physician for health-related concerns.

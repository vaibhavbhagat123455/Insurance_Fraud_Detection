# 🛡️ Insurance Fraud Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Arrays-013243?style=for-the-badge&logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

<br/>

> **An end-to-end Machine Learning web application that automatically classifies insurance claims as Fraudulent or Legitimate in real-time.**

<br/>

![Project Banner](https://img.shields.io/badge/Team_ID-EL2026TMID4684-1F4E79?style=flat-square)
![Accuracy](https://img.shields.io/badge/Model_Accuracy-~90%25-success?style=flat-square)
![Models](https://img.shields.io/badge/Models_Compared-6-blue?style=flat-square)
![Features](https://img.shields.io/badge/Input_Features-30+-orange?style=flat-square)

</div>

---

## 📽️ Demo Video

<div align="center">

<!-- 
  ✅ HOW TO ADD YOUR DEMO VIDEO:

  OPTION 1 — YouTube (Recommended):
  1. Upload your demo video to YouTube
  2. Replace YOUR_VIDEO_ID below with your YouTube video ID
  3. Uncomment the line below

  [![Watch Demo](https://img.youtube.com/vi/YOUR_VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)

  OPTION 2 — GitHub (for short videos under 100MB):
  1. Go to your GitHub repo → Issues → New Issue
  2. Drag and drop your video file into the comment box
  3. Copy the generated URL and paste it below

  ![Demo Video](YOUR_GITHUB_VIDEO_URL)

  OPTION 3 — Google Drive:
  1. Upload video to Google Drive → Share → Anyone with link
  2. Replace YOUR_FILE_ID with your file ID from the share link

  [![Watch Demo](https://drive.google.com/thumbnail?id=YOUR_FILE_ID)](https://drive.google.com/file/d/YOUR_FILE_ID/view)
-->
[![Watch Demo](demo_preview.png)](https://drive.google.com/file/d/18O2xrI04LSlY1sLCcQoVXIfWGBcBs09H/view?usp=sharing)
### 🎬 Click Here to Watch the Full Project Demo

</div>

---

## 📋 Table of Contents

- [About the Project](#-about-the-project)
- [Demo Video](#️-demo-video)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [ML Pipeline](#-ml-pipeline)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Installation & Setup](#️-installation--setup)
- [How to Run](#-how-to-run)
- [How It Works](#-how-it-works)
- [Screenshots](#-screenshots)
- [API Reference](#-api-reference)
- [Future Enhancements](#-future-enhancements)
- [Team](#-team)
- [License](#-license)

---

## 🎯 About the Project

The **Insurance Fraud Detection System** is a complete, production-ready machine learning application designed to combat the global insurance fraud problem — estimated to cost the industry **$40–80 billion annually**.

The system analyzes historical insurance claim data using multiple machine learning algorithms and predicts in real-time whether a submitted claim is **legitimate or fraudulent**. Users interact with a clean Flask web interface — no technical knowledge required.

### 🔍 Problem Statement

> Insurance companies process thousands of claims daily, yet **manual fraud review misses up to 30%** of fraudulent cases due to human bias, inconsistency, and the sheer volume of submissions. Rule-based systems quickly become obsolete as fraud patterns evolve.

### ✅ Our Solution

An automated ML-powered pipeline that:
- Preprocesses raw claim data with **Label Encoding + Standard Scaling**
- Trains and benchmarks **6 classification algorithms**
- Deploys the **best model (Decision Tree, ~90% accuracy)** via a **Flask web app**
- Returns **instant fraud/legitimate predictions** from 30+ input features

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **Multi-Model Benchmarking** | Trains & compares 6 ML models — picks the best objectively |
| ⚡ **Real-Time Prediction** | Sub-second fraud/legitimate verdict via Flask |
| 🌐 **Web Interface** | Clean HTML form accessible on any browser |
| 🔄 **Full ML Pipeline** | From raw CSV → preprocessing → training → deployment |
| 💾 **Model Persistence** | Saved with `pickle` — no retraining on restart |
| 📊 **30+ Features** | Policy, customer, and incident data analyzed |
| 🔒 **Input Validation** | Type-safe, validated form handling |

---

## 🛠️ Tech Stack

```
Language    →  Python 3.8+
Framework   →  Flask 2.x
ML Library  →  Scikit-Learn
Data        →  Pandas, NumPy
Visuals     →  Matplotlib, Seaborn
Frontend    →  HTML5, CSS3
Deployment  →  Pickle (model persistence)
```

---

## 📊 Dataset

| Attribute | Details |
|---|---|
| **File** | `insurance_claims.csv` |
| **Records** | 1,000 rows |
| **Total Columns** | 40 features + 1 target |
| **Target Variable** | `fraud_reported` (Y = Fraud, N = Legitimate) |
| **Class Distribution** | ~24.7% Fraud · ~75.3% Legitimate |

### 📌 Features Used (After Preprocessing — 30 Features)

```
months_as_customer    policy_number         policy_bind_date      policy_state
policy_csl            policy_deductable     policy_annual_premium insured_zip
insured_sex           insured_occupation    insured_hobbies       insured_relationship
capital_gains         capital_loss          incident_date         incident_type
collision_type        incident_severity     authorities_contacted incident_location
incident_hour_of_day  num_vehicles_involved property_damage       bodily_injuries
witnesses             police_report_avail   total_claim_amount    auto_make
auto_model            auto_year
```

### ❌ Removed Columns (Low Importance)

```
_c39  ·  age  ·  umbrella_limit  ·  insured_education_level  ·  incident_state
incident_city  ·  injury_claim  ·  property_claim  ·  vehicle_claim
```

---

## 🔬 ML Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PHASE                                  │
│                                                                         │
│  insurance_claims.csv                                                   │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐                │
│  │  Load Data  │ →  │Label Encoding│ →  │Drop 9 Columns│                │
│  │  (pandas)   │    │(categorical) │    │(low relevance│                │
│  └─────────────┘    └──────────────┘    └──────────────┘                │
│         │                                                               │
│         ▼                                                               │
│  ┌──────────────┐   ┌──────────────┐    ┌─────────────────────────┐     │
│  │  Train/Test  │ → │StandardScaler│ →  │  Train 6 ML Models      │     │
│  │  80% / 20%   │   │  (normalize) │    │  + Cross Validation     │     │
│  └──────────────┘   └──────────────┘    └─────────────────────────┘     │
│                                                  │                      │
│                                                  ▼                      │
│                                         Best Model Selected             │
│                                         (Decision Tree ~90%)            │
│                                                  │                      │
│                                    ┌─────────────┴───────────────────┐  │
│                                    │       Save with pickle          │  │
│                                    | dtc_model.pkl   std_scaler.pkl  │  │
│                                    └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         PREDICTION PHASE                                │
│                                                                         │
│  User Input (HTML Form)                                                 │
│         │                                                               │
│         ▼                                                               │
│  Flask app.py → Parse 30 fields → std_scaler.pkl → dtc_model.pkl        │
│                                                          │              │
│                                              ┌───────────┴───────────┐  │
│                                              │   Prediction = 0      │  │
│                                              │  Legal Insurance Claim│  │
│                                              └───────────────────────┘  │
│                                              ┌───────────────────────┐  │
│                                              │   Prediction = 1      │  │
│                                              │  Fraud Insurance Claim│  │
│                                              └───────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📈 Model Performance

| Model | Test Accuracy | Cross-Val | Status |
|---|---|---|---|
| **Decision Tree** | **~90.4%** | **~89.1%** | ✅ **Selected** |
| Random Forest | ~88.1% | ~87.0% | — |
| Support Vector Machine | ~85.3% | ~84.2% | — |
| K-Nearest Neighbors | ~83.2% | ~82.0% | — |
| Logistic Regression | ~80.5% | ~79.4% | — |
| Gaussian Naive Bayes | ~77.6% | ~76.3% | — |

### Confusion Matrix (Decision Tree)

```
                  Predicted: Legit    Predicted: Fraud
Actual: Legit          148                  9
Actual: Fraud           14                 29
```

| Metric | Score |
|---|---|
| Accuracy | ~90.4% |
| Precision | ~88.6% |
| Recall | ~86.2% |
| F1 Score | ~87.4% |

---

## 📁 Project Structure

```
insurance-fraud-detection/
│
├── 📄 app.py                     # Flask application — main entry point
├── 📊 insurance_claims.csv       # Raw dataset (1000 records, 40 features)
├── 📓 model_train.ipynb          # ML training notebook
├── 📓 Data_read.ipynb            # EDA & data exploration notebook
│
├── 🤖 dtc_model.pkl              # Trained Decision Tree model (generated)
├── 📐 std_scaler.pkl             # Fitted StandardScaler (generated)
│
├── 📋 requirements.txt           # Python dependencies
│
├── 🌐 templates/
│   ├── index.html                # Main prediction form (30+ inputs)
│   └── result.html               # Prediction result display page
│
└── 🎨 static/
    └── style.css                 # CSS styling
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python **3.8+** installed
- `pip` package manager
- Git

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/vaibhavbhagat123455/Insurance_Fraud_Detection.git
cd insurance-fraud-detection
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn flask jupyter
```

### 4️⃣ Train the Model

Run the training notebook to generate the `.pkl` files:

```bash
jupyter notebook model_train.ipynb
# Run all cells — this generates dtc_model.pkl and std_scaler.pkl
```

---

## 🚀 How to Run

```bash
python app.py
```

Then open your browser and navigate to:

```
http://localhost:5000
```

✅ You should see the Insurance Fraud Detection web form.

---

## 🔍 How It Works

### Step-by-Step Flow

```
1. User opens http://localhost:5000 in browser
        ↓
2. Fills in the HTML form with 30 insurance claim fields
        ↓
3. Clicks "Predict" — sends HTTP POST to /predict
        ↓
4. Flask app.py parses form values → converts to numeric array
        ↓
5. std_scaler.pkl normalizes the input array
        ↓
6. dtc_model.pkl (Decision Tree) makes prediction
        ↓
7. Result displayed:
   ✅  Prediction = 0  →  "Legal Insurance Claim"
   🚨  Prediction = 1  →  "Fraud Insurance Claim"
```

### Input Fields Explained

| Category | Fields |
|---|---|
| **Customer Info** | months_as_customer, insured_sex, insured_occupation, insured_hobbies, insured_relationship |
| **Policy Details** | policy_number, policy_bind_date, policy_state, policy_csl, policy_deductable, policy_annual_premium, insured_zip |
| **Financial** | capital_gains, capital_loss, total_claim_amount |
| **Incident Info** | incident_date, incident_type, collision_type, incident_severity, authorities_contacted, incident_location, incident_hour_of_the_day |
| **Scene Details** | number_of_vehicles_involved, property_damage, bodily_injuries, witnesses, police_report_available |
| **Vehicle Info** | auto_make, auto_model, auto_year |

---

## 🔌 API Reference

### `GET /`
Returns the main prediction form page.

**Response:** `index.html`

---

### `POST /predict`
Accepts claim data and returns a fraud prediction.

**Request:** `multipart/form-data`

| Parameter | Type | Description |
|---|---|---|
| `months_as_customer` | `float` | Duration as customer in months |
| `policy_number` | `float` | Encoded policy number |
| `incident_type` | `float` | Encoded incident type |
| `total_claim_amount` | `float` | Total claim value in USD |
| *(+ 26 more fields)* | `float` | See Input Fields section above |

**Response:** `index.html` with `prediction_text`

```python
# Prediction = 0
prediction_text = "Legal Insurance Claim"

# Prediction = 1
prediction_text = "Fraud Insurance Claim"
```

---

## 🔮 Future Enhancements

- [ ] 🧠 Upgrade to **XGBoost / LightGBM** for higher accuracy
- [ ] ⚖️ Handle class imbalance with **SMOTE oversampling**
- [ ] 🔍 Add **SHAP / LIME** explainability for each prediction
- [ ] 🗄️ Add **PostgreSQL** to log all predictions and audit trail
- [ ] 🔐 Add **user authentication** and role-based access control
- [ ] 📱 Build **React Native** mobile app for field adjusters
- [ ] ☁️ Deploy on **AWS EC2 / Heroku** with Gunicorn + Nginx
- [ ] 🔄 Implement **MLflow** for model versioning and retraining
- [ ] 📊 Add **live dashboard** with fraud statistics and charts
- [ ] 🌐 Expose **REST JSON API** for third-party CRM integration

---

## 👥 Team

| Detail | Info |
|---|---|
| **Team ID** | EL2026TMID4684 |
| **Project** | Insurance Fraud Detection System |
| **Phase** | Project Design Phase-I |
| **Domain** | Machine Learning / AI |

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License — Free to use, modify, and distribute with attribution.
```

---

## 🙏 Acknowledgements

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Kaggle — Insurance Fraud Dataset](https://www.kaggle.com/)

---

<div align="center">

**⭐ If this project helped you, please give it a star!**

Made with ❤️ | Team EL2026TMID4684

</div>

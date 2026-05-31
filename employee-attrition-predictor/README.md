# 👥 Employee Attrition Predictor

An ML-powered HR analytics tool that predicts whether an employee is likely to leave the company using a Random Forest classifier trained on real IBM HR data.

🔗 **Live Demo:** https://attrition-ai-nkaapouh8ynghxwmnx5dzv.streamlit.app/

---

## Features

- 🔮 Real-time attrition prediction — input employee details and get instant risk assessment
- 📊 Data insights dashboard — attrition breakdown by age, department, and key factors
- 🎯 88% model accuracy using Random Forest classifier
- 📈 Top 10 attrition factors visualised with feature importance chart
- ☁️ Deployed live on Streamlit Cloud

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.14 |
| Machine Learning | Scikit-learn, Random Forest |
| Data Processing | Pandas, NumPy |
| Visualisation | Plotly Express |
| Web App | Streamlit |
| Model Persistence | Joblib |
| Deployment | Streamlit Cloud |

---

## Key Findings from the Data

- 💰 Monthly Income is the #1 factor — low salary drives attrition most
- ⏰ Overtime workers are significantly more likely to leave
- 👴 Younger employees (age 25-35) have highest attrition rate
- 📍 Employees living far from office leave more frequently
- 🏢 Sales department has highest attrition rate

---

## ML Model Details

- **Algorithm:** Random Forest Classifier
- **Dataset:** IBM HR Analytics (1,470 employees, 35 features)
- **Accuracy:** 88.10%
- **Train/Test Split:** 80/20

---

## Run Locally

```bash
git clone https://github.com/mohithjain0678-sudo/attrition-ai.git
cd attrition-ai/employee-attrition-predictor
pip install -r requirements.txt
streamlit run app.py
```

---

## Dataset

IBM HR Analytics Employee Attrition Dataset from Kaggle.

---

## Author

**Mohith** — ECE Student at VIT Chennai

🔗 [LinkedIn](https://linkedin.com/in/mohithjain-302076397) | [GitHub](https://github.com/mohithjain0678-sudo)
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(page_title="Employee Attrition Predictor", page_icon="👥", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, 'notebooks/attrition_model.pkl'))
feature_columns = joblib.load(os.path.join(BASE_DIR, 'notebooks/feature_columns.pkl'))
df = pd.read_csv(os.path.join(BASE_DIR, 'data/WA_Fn-UseC_-HR-Employee-Attrition.csv'))

st.title("👥 Employee Attrition Predictor")
st.markdown("AI-powered HR analytics tool to predict employee attrition using Machine Learning")

tab1, tab2 = st.tabs(["🔮 Predict Attrition", "📊 Data Insights"])

with tab1:
    st.sidebar.header("Enter Employee Details")
    age = st.sidebar.slider("Age", 18, 60, 30)
    monthly_income = st.sidebar.number_input("Monthly Income ($)", 1000, 20000, 5000)
    overtime = st.sidebar.selectbox("Works Overtime?", ["Yes", "No"])
    distance = st.sidebar.slider("Distance From Home (km)", 1, 30, 10)
    job_satisfaction = st.sidebar.slider("Job Satisfaction (1-4)", 1, 4, 3)
    years_at_company = st.sidebar.slider("Years at Company", 0, 40, 5)
    total_working_years = st.sidebar.slider("Total Working Years", 0, 40, 8)
    num_companies = st.sidebar.slider("Number of Companies Worked", 0, 9, 2)
    work_life_balance = st.sidebar.slider("Work Life Balance (1-4)", 1, 4, 3)
    environment_satisfaction = st.sidebar.slider("Environment Satisfaction (1-4)", 1, 4, 3)

    input_data = {}
    for col in feature_columns:
        input_data[col] = 0

    input_data['Age'] = age
    input_data['MonthlyIncome'] = monthly_income
    input_data['OverTime'] = 1 if overtime == "Yes" else 0
    input_data['DistanceFromHome'] = distance
    input_data['JobSatisfaction'] = job_satisfaction
    input_data['YearsAtCompany'] = years_at_company
    input_data['TotalWorkingYears'] = total_working_years
    input_data['NumCompaniesWorked'] = num_companies
    input_data['WorkLifeBalance'] = work_life_balance
    input_data['EnvironmentSatisfaction'] = environment_satisfaction

    input_df = pd.DataFrame([input_data])

    col1, col2, col3 = st.columns(3)
    col1.metric("Age", age)
    col2.metric("Monthly Income", f"${monthly_income:,}")
    col3.metric("Years at Company", years_at_company)

    st.divider()

    if st.button("🔮 Predict Attrition", use_container_width=True):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        if prediction == 1:
            st.error("⚠️ HIGH RISK — This employee is likely to LEAVE")
            st.metric("Probability of Leaving", f"{probability[1]*100:.1f}%")
        else:
            st.success("✅ LOW RISK — This employee is likely to STAY")
            st.metric("Probability of Staying", f"{probability[0]*100:.1f}%")

        st.divider()
        st.subheader("📊 Key Risk Factors")
        factors = {
            "Overtime": "High Risk 🔴" if overtime == "Yes" else "Low Risk 🟢",
            "Job Satisfaction": "High Risk 🔴" if job_satisfaction <= 2 else "Low Risk 🟢",
            "Work Life Balance": "High Risk 🔴" if work_life_balance <= 2 else "Low Risk 🟢",
            "Distance from Home": "High Risk 🔴" if distance > 20 else "Low Risk 🟢",
            "Environment Satisfaction": "High Risk 🔴" if environment_satisfaction <= 2 else "Low Risk 🟢"
        }
        for factor, risk in factors.items():
            st.write(f"**{factor}:** {risk}")

with tab2:
    st.subheader("📈 Company Attrition Overview")

    col1, col2, col3 = st.columns(3)
    total = len(df)
    left = len(df[df['Attrition'] == 'Yes'])
    stayed = len(df[df['Attrition'] == 'No'])
    col1.metric("Total Employees", total)
    col2.metric("Employees Left", left)
    col3.metric("Attrition Rate", f"{left/total*100:.1f}%")

    st.divider()

    attrition_counts = df['Attrition'].value_counts().reset_index()
    attrition_counts.columns = ['Attrition', 'Count']
    fig1 = px.pie(attrition_counts, values='Count', names='Attrition',
                  title='Overall Attrition Distribution',
                  color_discrete_sequence=['#00CC96', '#EF553B'])
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(df, x='Age', color='Attrition',
                        title='Attrition by Age Group',
                        barmode='overlay',
                        color_discrete_sequence=['#00CC96', '#EF553B'])
    st.plotly_chart(fig2, use_container_width=True)

    dept_attrition = df[df['Attrition'] == 'Yes']['Department'].value_counts().reset_index()
    dept_attrition.columns = ['Department', 'Count']
    fig3 = px.bar(dept_attrition, x='Department', y='Count',
                  title='Attrition by Department',
                  color='Count', color_continuous_scale='Reds')
    st.plotly_chart(fig3, use_container_width=True)

    feature_imp = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)
    fig4 = px.bar(feature_imp, x='Importance', y='Feature', orientation='h',
                  title='Top 10 Factors Causing Attrition',
                  color='Importance', color_continuous_scale='Blues')
    fig4.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig4, use_container_width=True)
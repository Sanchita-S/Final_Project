# Final_Project
# 📊 Job Salary Prediction Dashboard

This project is an end-to-end data application that collects real-time job data from the **Adzuna API**, cleans and processes it, trains a machine learning model to predict salaries, and visualizes the insights in a sleek, multi-tabbed **Streamlit dashboard**.

## 🚀 Features
- **Job Data Collection:**
  - Fetches job postings from the Adzuna API (Canada + USA)
  - Parses title, company, location, salary, and job URL
- **Data Cleaning & Preparation:**
  - Drops duplicates and nulls
  - Computes average salary
  - Extracts job seniority and encodes location
- **Salary Prediction Model:**
  - Uses `RandomForestRegressor` from scikit-learn
  - Trained on job title, seniority level, and location
  - Evaluates model with MAE and R2 score
- **Streamlit Dashboard:**
  - Clean dark theme with professional styling
  - Interactive sidebar filters (company, location)
  - Tabs for:
    - Dashboard Overview
    - Error Distribution
    - Top Hiring Companies
    - Job Listings Table

## 📂 Folder Structure
```
📁 Final_Project/
├── dd_salary.py                     # Main Streamlit dashboard script
├── full_salary_predictionss.csv # Cleaned + predicted job data
├── search.jpg                 # Sidebar image
├── screen.docx                # Screenshots of results
├──final_project_st.ppt
└── README.md                  # Project documentation (this file)
```

## 🧪 How to Run
1. Clone the repository
```bash
git clone https://github.com/yourusername/job-salary-dashboard.git
cd job-salary-dashboard
```
2. Install dependencies (recommended in a virtual environment)
```bash

```
3. Launch the dashboard
```bash
streamlit run dash.py
```

## 🔍 Requirements
- Python 3.8+
- Streamlit
- Pandas
- Plotly
- scikit-learn

Install them with:
```bash
pip install streamlit pandas plotly scikit-learn
```

## 📊 Sample Output
- Dynamic salary charts by role
- Predicted vs Actual salary line plot
- Top 10 hiring companies
- Interactive job table with URLs

## 🌐 Data Source
- Adzuna Job Search API (https://developer.adzuna.com/)

## 🙌 Contributors
- Built by Sanchita
- Powered by Python, Streamlit, and 💡 curiosity




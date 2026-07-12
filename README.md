# COVID Drug Effect Analysis Using Causal Inference

## Project Overview
A data engineering and analytics project analyzing 10,000+ patient records 
to evaluate COVID-19 treatment effectiveness using causal inference methods, 
machine learning models, and statistical analysis — generating actionable 
clinical insights to support data-driven healthcare decisions.

## Problem Statement
Healthcare teams needed reliable, data-driven insights into which treatments 
were most effective for COVID-19 patients — but raw clinical data was:
- Unstructured and inconsistent across multiple sources
- Missing critical variables affecting treatment outcomes
- Difficult to analyze without proper data engineering workflows
- Lacking standardized reporting for clinical decision making

## Solution
Built a complete data engineering and analytics pipeline:
- Data ingestion and cleaning from clinical records
- Feature engineering and variable validation
- Multiple ML model comparison for outcome prediction
- Analytical reporting and visualization across patient groups

## Pipeline Flow
Raw Clinical Data → Data Cleaning → Feature Engineering → 
Model Training → Evaluation → Analytical Reports & Insights

## Technologies Used
- Programming: Python, SQL
- Machine Learning: XGBoost, Logistic Regression, Naive Bayes, Gradient Boosting
- Data Processing: Pandas, NumPy, Scikit-Learn
- Visualization: Matplotlib, Seaborn, Power BI
- Database: PostgreSQL, MySQL
- Development: Jupyter Notebook, Git

## Project Structure
- data/ — Sample dataset and data dictionary
- sql/ — Data extraction and transformation queries
- notebooks/ — Jupyter notebooks for EDA and modeling
- src/ — Python scripts for data processing and modeling
- reports/ — Analytical reports and visualizations
- docs/ — Project documentation

## Key Results
- Analyzed 10,000+ patient records across multiple treatment groups
- Achieved reliable outcome predictions using XGBoost as best performing model
- Identified key factors influencing COVID-19 recovery outcomes
- Generated analytical reports comparing treatment performance across patient groups
- Reduced dataset preparation effort through automated preprocessing pipelines
- Delivered actionable insights supporting clinical decision making

## Machine Learning Models Compared

| Model | Purpose | Performance |
|---|---|---|
| Logistic Regression | Baseline classification | Good interpretability |
| Naive Bayes | Probabilistic classification | Fast training |
| Gradient Boosting | Ensemble method | High accuracy |
| XGBoost | Advanced boosting | Best performance |

## Data Engineering Highlights
- Automated data ingestion and cleaning pipeline using Python
- Feature engineering on 20+ clinical variables
- Data validation checks ensuring data quality and consistency
- SQL-based data extraction and transformation workflows
- Automated reporting pipeline reducing manual effort significantly

## Future Improvements
- Migrate pipeline to cloud-native architecture using AWS or Azure
- Implement Apache Airflow for automated pipeline orchestration
- Add real-time data ingestion using streaming technologies
- Build interactive Power BI dashboard for clinical reporting
- Implement Great Expectations for automated data validation
- Add CI/CD pipeline for automated model retraining

## Author
Prathap Ramachandra
Data Engineer | Python · SQL · PySpark · Snowflake · AWS · Azure · Databricks
Email: prathapramachandra2000@gmail.com
GitHub: https://github.com/prathapramachandra
LinkedIn: https://www.linkedin.com/in/prathapramachandra

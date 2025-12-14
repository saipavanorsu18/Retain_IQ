# Retain_IQ
Customer Churn Intelligence Hub is an end-to-end, production-style AutoML web application that enables companies from any domain (telecom, banking, insurance, SaaS, etc.) to analyze churn patterns and predict future churn using their own datasets.
Customer Churn Intelligence Hub

A universal, end-to-end AutoML web application that allows companies to analyze, train, and predict customer churn using any dataset, regardless of industry or column structure.

This project transforms raw business data into actionable churn insights through an interactive, human-friendly dashboard built with Streamlit and Machine Learning.

🚀 Project Overview

Customer churn is a critical problem across industries such as telecom, banking, insurance, and SaaS.
This application enables organizations to:

Upload historical customer data

Automatically train the best-performing ML model

Visualize churn patterns and feature importance

Predict churn for new, unseen customers

Download prediction reports for business decision-making

Unlike traditional churn systems that rely on fixed schemas or pre-trained models, this project implements a dynamic AutoML pipeline that adapts to any dataset structure.

🧠 Key Features
🔹 Universal AutoML Pipeline

Works with any company dataset

User selects the target column (e.g., Churn, Exited)

Automatically handles:

Categorical encoding

Train–test splitting

Model training & evaluation

🔹 Multi-Model Training & Selection

Trains multiple models:

Random Forest

Gradient Boosting

AdaBoost

Logistic Regression

Automatically selects the best model based on performance

🔹 Interactive Analytics Dashboard

KPI cards (Accuracy, Recall, F1 Score)

Confusion Matrix visualization

Feature Importance analysis

Churn distribution charts

Correlation heatmaps

Probability-based churn predictions

🔹 Real-World Prediction Workflow

Train on historical data (with known churn)

Predict churn for new customers without labels

Generate churn probability scores

Download results as CSV

🔹 Persistent State Management

Uploaded datasets remain available across pages

Trained model persists during the session

Seamless navigation between upload, training, and prediction

🖥️ Tech Stack

Frontend / UI: Streamlit

Data Processing: Pandas, NumPy

Machine Learning: Scikit-learn

Visualization: Plotly, Seaborn, Matplotlib

Model Persistence: Joblib

📌 Why This Project Stands Out

✅ Not limited to a single industry
✅ No hard-coded features or schemas
✅ Fully automated ML pipeline
✅ Clean, human-friendly UI
✅ Business-ready predictions and reports

This project bridges the gap between machine learning experimentation and real-world deployment.

🎯 Use Cases

Telecom churn analysis

Bank customer exit prediction

Subscription-based business retention

SaaS user behavior analysis

Academic and portfolio projects

📥 How to Run
pip install -r requirements.txt
streamlit run churn_intelligence_hub.py

🌟 Future Enhancements

Hyperparameter tuning

Model explainability (SHAP values)

Role-based dashboards

Cloud deployment (Streamlit Cloud / HuggingFace)

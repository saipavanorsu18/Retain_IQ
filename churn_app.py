import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import io

# 🌈 Streamlit page setup
st.set_page_config(page_title="RetainIQ", layout="wide")

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to bottom right, #f9fafb, #e3f2fd);
        color: #1a1a1a;
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        border-radius: 20px;
        padding: 20px;
        background: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
    }
    .title {
        font-size: 50px !important;
        font-weight: 800 !important;
        color: #1e88e5 !important;
        text-align: center !important;
        margin-bottom: 0 !important;
        text-shadow: 0 0 20px  
        rgba(30,142,133,0.5) !important;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #555;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<p class='title'>RetainIQ💡</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload. Train. Analyze. Predict. — An Intelligent Dashboard for your Company Dataset.</p>", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("🧭 Navigation")
section = st.sidebar.radio("Go to Section:", ["📂 Data Upload", "📊 Dashboard & Training", "🧠 Prediction", "📁 Model Management"])

# --- Upload Section ---
if section == "📂 Data Upload":
    st.header("📥 Upload Your Datasets")

    # Load previous data if available
    if 'train_df' in st.session_state:
        st.success("✅ Previously uploaded training dataset found.")
        st.dataframe(st.session_state['train_df'].head())

    if 'predict_df' in st.session_state:
        st.success("✅ Previously uploaded prediction dataset found.")
        st.dataframe(st.session_state['predict_df'].head())

    train_file = st.file_uploader("Upload Training Dataset (must include target column)", type=["csv"], key="train_upload")
    predict_file = st.file_uploader("Upload New Dataset (for churn prediction)", type=["csv"], key="predict_upload")

    if train_file is not None:
        st.session_state['train_df'] = pd.read_csv(train_file)
        st.success("✅ Training dataset uploaded successfully!")
        st.dataframe(st.session_state['train_df'].head())

    if predict_file is not None:
        st.session_state['predict_df'] = pd.read_csv(predict_file)
        st.success("✅ Prediction dataset uploaded successfully!")
        st.dataframe(st.session_state['predict_df'].head())
if st.button("🧹 Reset All Data"):
    st.session_state.clear()
    st.success("Session cleared! You can start fresh.")


# --- Dashboard Section ---
elif section == "📊 Dashboard & Training":
    st.header("📈 Data Analysis & Model Training")

    if 'train_df' not in st.session_state:
        st.warning("⚠️ Please upload a training dataset first.")
    else:
        df = st.session_state['train_df']
        target_col = st.selectbox("🎯 Select Target Column", df.columns)

        if target_col:
            X = df.drop(columns=[target_col])
            y = df[target_col]

            # Encode categorical data
            X_encoded = X.copy()
            encoders = {}
            for col in X_encoded.columns:
                if X_encoded[col].dtype == 'object':
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                    encoders[col] = le

            if y.dtype == 'object':
                y = LabelEncoder().fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

            # --- Model Training ---
            st.subheader("🤖 AutoML: Model Training & Selection")

            models = {
                "Random Forest": RandomForestClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "AdaBoost": AdaBoostClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=1000)
            }

            results = []
            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                prec = precision_score(y_test, preds, average='weighted', zero_division=0)
                rec = recall_score(y_test, preds, average='weighted', zero_division=0)
                f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
                results.append((name, acc, prec, rec, f1))

            result_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])
            st.dataframe(result_df.style.highlight_max(color='#90caf9', axis=0))

            # --- Select best model ---
            best_model_name = result_df.sort_values(by="Accuracy", ascending=False).iloc[0]["Model"]
            best_model = models[best_model_name]
            st.success(f"🏆 Best Performing Model: **{best_model_name}**")

            # --- KPIs ---
            st.markdown("### 📊 Key Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{result_df['Accuracy'].max():.2%}")
            with col2:
                st.metric("Recall", f"{result_df['Recall'].max():.2%}")
            with col3:
                st.metric("F1 Score", f"{result_df['F1 Score'].max():.2%}")

            # --- Confusion Matrix ---
            cm = confusion_matrix(y_test, best_model.predict(X_test))
            fig = px.imshow(cm, text_auto=True, title="Confusion Matrix", color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)

            # --- Feature Importance ---
            if hasattr(best_model, 'feature_importances_'):
                feat_imp = pd.DataFrame({
                    'Feature': X_encoded.columns,
                    'Importance': best_model.feature_importances_
                }).sort_values(by='Importance', ascending=True)
                fig2 = px.bar(feat_imp, x='Importance', y='Feature', orientation='h',
                              title="Feature Importance", color='Importance', color_continuous_scale='viridis')
                st.plotly_chart(fig2, use_container_width=True)

            # --- Churn Distribution ---
            fig3 = px.pie(df, names=target_col, title="Churn Distribution", color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig3, use_container_width=True)

            # Save best model to session
            st.session_state['best_model'] = best_model

            if st.button("💾 Save Best Model"):
                joblib.dump(best_model, "best_churn_model.pkl")
                st.success("✅ Model saved and stored in memory!")


# --- Prediction Section ---
elif section == "🧠 Prediction":
    st.header("🔮 Predict Churn for New Customers")

    if 'predict_df' not in st.session_state:
        st.warning("⚠️ Please upload a prediction dataset first.")
    else:
        if 'best_model' not in st.session_state:
            st.warning("⚠️ Please train a model first in the Dashboard & Training section.")
        else:
            model = st.session_state['best_model']

            new_df = st.session_state['predict_df']

            # Encode and align columns
            new_encoded = new_df.copy()
            for col in new_encoded.columns:
                if new_encoded[col].dtype == 'object':
                    new_encoded[col] = LabelEncoder().fit_transform(new_encoded[col].astype(str))

            # Dummy align columns
            for c in model.feature_names_in_:
                if c not in new_encoded.columns:
                    new_encoded[c] = 0
            new_encoded = new_encoded[model.feature_names_in_]

            preds = model.predict(new_encoded)
            probs = model.predict_proba(new_encoded)[:, 1]

            new_df["Predicted_Churn"] = preds
            new_df["Churn_Probability"] = probs

            st.dataframe(new_df.head())

            fig4 = px.histogram(new_df, x="Churn_Probability", nbins=20, title="Predicted Churn Probability Distribution")
            st.plotly_chart(fig4, use_container_width=True)

            csv = new_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Prediction Results", csv, "churn_predictions.csv")

# --- Model Management ---
elif section == "📁 Model Management":
    st.header("🗂️ Manage Saved Models")
    st.write("Here you can upload or load existing trained models to reuse for prediction.")

    uploaded_model = st.file_uploader("Upload a Trained Model (.pkl)", type=["pkl"])
    if uploaded_model:
        model = joblib.load(uploaded_model)
        st.success("✅ Model loaded successfully and ready to use.")
 
st.markdown("<p class='subtitle'>© 2025 Sai Pavan.All Rights Reserved.</p>", unsafe_allow_html=True)
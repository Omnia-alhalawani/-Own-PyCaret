import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import setup as cls_setup, compare_models as cls_compare, create_model as cls_create, pull as cls_pull, save_model as cls_save
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, create_model as reg_create, pull as reg_pull, save_model as reg_save
from sklearn.preprocessing import LabelEncoder, FeatureHasher

# Streamlit App
st.title("AutoML with PyCaret")

# Upload Data
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data:")
    st.dataframe(df.head())

    # Drop Columns
    drop_cols = st.multiselect("Select columns to drop (optional)", df.columns)
    df.drop(columns=drop_cols, inplace=True)
    
    # Handle Missing Values
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if cat_cols:
        cat_fill_method = st.radio("How to handle missing values for categorical columns?", ["Mode", "Additional Class"])
        for col in cat_cols:
            df[col].fillna(df[col].mode()[0] if cat_fill_method == "Mode" else "Missing", inplace=True)
    
    if num_cols:
        num_fill_method = st.radio("How to handle missing values for numerical columns?", ["Mean", "Median", "Mode"])
        for col in num_cols:
            if num_fill_method == "Mean":
                df[col].fillna(df[col].mean(), inplace=True)
            elif num_fill_method == "Median":
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Encoding method selection
    encoding_method = st.radio("Choose how to encode categorical data:", ["One-Hot Encoding", "Label Encoding", "Feature Hashing"])

    if st.button("🔄 Process Data"):
        if encoding_method == "One-Hot Encoding":
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        elif encoding_method == "Label Encoding":
            label_enc = LabelEncoder()
            for col in cat_cols:
                df[col] = label_enc.fit_transform(df[col])
        elif encoding_method == "Feature Hashing":
            fh = FeatureHasher(n_features=100, input_type="string")
            for col in cat_cols:
                hashed_features = fh.transform(df[col].astype(str))
                df_hashed = pd.DataFrame(hashed_features.toarray())
                df = pd.concat([df, df_hashed], axis=1).drop(columns=[col])

        st.write("✅ Processed Data Preview:")
        st.dataframe(df.head())
    
    # Select Target and Features
    y_col = st.selectbox("Select target variable (Y)", df.columns)
    X_cols = st.multiselect("Select feature columns (X)", [col for col in df.columns if col != y_col])
    
    # Detect Task Type
    if df[y_col].nunique() <= 10 and df[y_col].dtype == 'object':
        task_type = "classification"
        st.write("### Detected Task Type: Classification")
    else:
        task_type = "regression"
        st.write("### Detected Task Type: Regression")
    
    # Select Model
    available_models = ['Auto Select', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'SVM', 'KNN']
    selected_model = st.selectbox("Select a model to train", available_models)
    
    # Train Models with PyCaret
    if st.button("Train Models"):
        st.write("Training models...")
        
        if task_type == "classification":
            clf = cls_setup(data=df, target=y_col, session_id=42)
            if selected_model == 'Auto Select':
                best_model = cls_compare()
            else:
                model_map = {'Decision Tree': 'dt', 'Random Forest': 'rf', 'Gradient Boosting': 'gbc', 'SVM': 'svm', 'KNN': 'knn'}
                best_model = cls_create(model_map[selected_model])
            results = cls_pull()
        else:
            reg = reg_setup(data=df, target=y_col, session_id=42)
            if selected_model == 'Auto Select':
                best_model = reg_compare()
            else:
                model_map = {'Decision Tree': 'dt', 'Random Forest': 'rf', 'Gradient Boosting': 'gbr', 'SVM': 'svm', 'KNN': 'knn'}
                best_model = reg_create(model_map[selected_model])
            results = reg_pull()
        
        st.write("### Best Model:")
        st.write(best_model)
        
        st.write("### Model Evaluation Metrics:")
        st.dataframe(results)
        
        if st.button("Save Model"):
            if task_type == "classification":
                cls_save(best_model, "best_model")
            else:
                reg_save(best_model, "best_model")
            st.success("Model saved successfully!")
            with open("best_model.pkl", "rb") as f:
                st.download_button("Download Model", f, "best_model.pkl")


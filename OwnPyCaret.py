import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import *
from pycaret.regression import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Streamlit App
st.title("AutoML with PyCaret")

#  Upload Data
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data:")
    st.dataframe(df.head())

    # Drop Columns
    drop_cols = st.multiselect("Select columns to drop (optional)", df.columns)
    df.drop(columns=drop_cols, inplace=True)
    
    # EDA
    if st.checkbox("Perform EDA?"):
        eda_cols = st.multiselect("Select columns for analysis", df.columns)
        if eda_cols:
            st.write("### EDA Summary:")
            st.write(df[eda_cols].describe())
    
    #Handle Missing Values
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if categorical_cols:
        cat_fill_method = st.radio("How to handle missing values for categorical columns?", ["Mode", "Additional Class"])
        for col in categorical_cols:
            if cat_fill_method == "Mode":
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna("Missing", inplace=True)
    
    if num_cols:
        num_fill_method = st.radio("How to handle missing values for continuous columns?", ["Mean", "Median", "Mode"])
        for col in num_cols:
            if num_fill_method == "Mean":
                df[col].fillna(df[col].mean(), inplace=True)
            elif num_fill_method == "Median":
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Encoding method selection
    encoding_method = st.radio(
        "Choose how to encode categorical data:",
        ["One-Hot Encoding", "Label Encoding"]
    )

    if st.button("🔄 Process Data"):
        if encoding_method == "One-Hot Encoding":
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        elif encoding_method == "Label Encoding":
            label_enc = LabelEncoder()
            for col in categorical_cols:
                df[col] = label_enc.fit_transform(df[col])

        st.write("✅ Processed Data Preview:")
        st.write(df.head())
    
    #  Select X (Features) and Y (Target)
    y_col = st.selectbox("Select target variable (Y)", df.columns)
    X_cols = st.multiselect("Select feature columns (X)", [col for col in df.columns if col != y_col])
    X = df[X_cols]
    y = df[y_col]
    
    #  Detect Task Type
    if df[y_col].nunique() <= 10 and df[y_col].dtype == 'object':
        task_type = "classification"
        st.write("### Detected Task Type: Classification")
    else:
        task_type = "regression"
        st.write("### Detected Task Type: Regression")
    
    #  Select Model
    available_models = ['Auto Select', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'SVM', 'KNN']
    selected_model = st.selectbox("Select a model to train", available_models)
    
    # Train Models with PyCaret
    if st.button("Train Models"):
        st.write("Training models...")
        if task_type == "classification":
            clf = setup(data=df, target=y_col, session_id=42)
            if selected_model == 'Auto Select':
                best_model = compare_models()
            else:
                best_model = create_model(selected_model.lower().replace(" ", "_"))
        else:
            reg = setup(data=df, target=y_col, session_id=42)
            if selected_model == 'Auto Select':
                best_model = compare_models()
            else:
                best_model = create_model(selected_model.lower().replace(" ", "_"))
        
        st.write("### Best Model:")
        st.write(best_model)
        
        # Show Evaluation 
        st.write("### Model Evaluation Metrics:")
        results = pull()
        st.dataframe(results)
        
        # Save Model
        if st.button("Save Model"):
            save_model(best_model, "best_model")
            st.success("Model saved successfully!")
            
            with open("best_model.pkl", "rb") as f:
                st.download_button("Download Model", f, "best_model.pkl")


import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import *
from pycaret.regression import *
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
    
    # Step 4: Handle Missing Values
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if cat_cols:
        cat_fill_method = st.radio("How to handle missing values for categorical columns?", ["Mode", "Additional Class"])
        for col in cat_cols:
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
        ["One-Hot Encoding", "Label Encoding", "Feature Hashing"]
    )

    if st.button("🔄 Process Data"):
        if encoding_method == "One-Hot Encoding":
            # Apply One-Hot Encoding only to selected columns
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        elif encoding_method == "Label Encoding":
            label_enc = LabelEncoder()
            for col in categorical_cols:
                df[col] = label_enc.fit_transform(df[col])
        elif encoding_method == "Feature Hashing":
            fh = FeatureHasher(n_features=100, input_type="string")
            for col in categorical_cols:
                hashed_features = fh.transform(df[col].astype(str))
                df_hashed = pd.DataFrame(hashed_features.toarray())
                df = pd.concat([df, df_hashed], axis=1).drop(columns=[col])

        st.write("✅ Processed Data Preview:")
        st.write(df.head())
    
    # Step 6: Select X (Features) and Y (Target)
    y_col = st.selectbox("Select target variable (Y)", df.columns)
    X_cols = st.multiselect("Select feature columns (X)", [col for col in df.columns if col != y_col])
    X = df[X_cols]
    y = df[y_col]
    
    # Step 7: Detect Task Type
    if df[y_col].nunique() <= 10 and df[y_col].dtype == 'object':
        task_type = "classification"
        st.write("### Detected Task Type: Classification")
    else:
        task_type = "regression"
        st.write("### Detected Task Type: Regression")
    
    # Step 8: Select Model
    available_models = ['Auto Select', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'SVM', 'KNN']
    selected_model = st.selectbox("Select a model to train", available_models)
    
    # Step 9: Train Models with PyCaret
    if st.button("Train Models"):
        st.write("Training models...")
        if task_type == "classification":
            clf = setup(data=df, target=y_col, silent=True, session_id=42)
            if selected_model == 'Auto Select':
                best_model = compare_models()
            else:
                best_model = create_model(selected_model.lower().replace(" ", "_"))
        else:
            reg = setup(data=df, target=y_col, silent=True, session_id=42)
            if selected_model == 'Auto Select':
                best_model = compare_models()
            else:
                best_model = create_model(selected_model.lower().replace(" ", "_"))
        
        st.write("### Best Model:")
        st.write(best_model)
        
        # Step 10: Show Evaluation Metrics
        st.write("### Model Evaluation Metrics:")
        results = pull()
        st.dataframe(results)
        
        # Step 11: Save Model
        if st.button("Save Model"):
            save_model(best_model, "best_model")
            st.success("Model saved successfully!")
            
            with open("best_model.pkl", "rb") as f:
                st.download_button("Download Model", f, "best_model.pkl")

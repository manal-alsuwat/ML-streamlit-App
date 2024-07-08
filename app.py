import streamlit as st
from pycaret.regression import setup, compare_models, create_model, predict_model, pull, tune_model, plot_model

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pycaret.regression import *


from pycaret.regression import setup as reg_setup, compare_models as reg_compare_models, create_model as reg_create_model, predict_model as reg_predict_model, pull as reg_pull, tune_model as reg_tune_model
from pycaret.classification import setup as cls_setup, compare_models as cls_compare_models, create_model as cls_create_model, predict_model as cls_predict_model, pull as cls_pull, tune_model as cls_tune_model
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Set page title and icon
st.set_page_config(page_title='Machine Learning App', page_icon=':robot_face:')


def handle_categorical_encoding(df):
    st.subheader('Handle Categorical Data Encoding:')
    encoding_option = st.selectbox(
        'Choose how to encode categorical data:',
        ('None', 'One-Hot Encoding', 'Label Encoding')
    )
    if st.button('Apply Encoding'):

        if encoding_option == 'One-Hot Encoding':
            df = pd.get_dummies(df, drop_first=True)
            st.success('One-Hot Encoding applied!')
        elif encoding_option == 'Label Encoding':
            label_encoder = LabelEncoder()
            for column in df.select_dtypes(include=['object']).columns:
                df[column] = label_encoder.fit_transform(df[column])
            st.success('Label Encoding applied!')

        return df
    return df




def perform_eda(df):
    st.subheader('Perform Exploratory Data Analysis (EDA):')
    eda_option = st.selectbox(
        'Do you want to perform EDA?',
        ('Yes', 'No')
    )

    if eda_option == 'Yes':
        columns_to_analyze = st.multiselect(
            'Select columns to analyze:',
            df.columns
        )

        if st.button('Perform EDA'):
            if len(columns_to_analyze) > 0:
                eda_df = df[columns_to_analyze]
                st.write(eda_df.describe())
            else:
                st.warning('Please select at least one column to analyze.')
        return True
    return False





def handle_missing_values(df):
    st.subheader('Handle Missing Values:')
    missing_values_option = st.selectbox(
        'Choose how to handle missing values:',
        ('None', 'Drop rows with missing values', 'Impute missing values with a specific value')
    )

    if st.button('Handle Missing Values'):
        if missing_values_option == 'Drop rows with missing values':
            df = df.dropna()
            st.success('Rows with missing values dropped!')
        elif missing_values_option == 'Impute missing values with a specific value':
            fill_value = st.text_input('Enter the value to fill missing values with:')
            if st.button('Apply Fill Value'):
                df = df.fillna(fill_value)
                st.success(f'Missing values filled with {fill_value}!')
        else:
            st.info('You can implement advanced imputation techniques here.')
        return df
    return df


def main():
    st.title('Machine Learning Web App')

    # Upload CSV file
    uploaded_file = st.file_uploader('Upload CSV file', type=['csv'])
    if uploaded_file is not None:
        st.success('File uploaded successfully!')
        df = pd.read_csv(uploaded_file)

        # Display uploaded data
        st.subheader('Uploaded Data:')
        st.write(df.head())

        # Ask user if they want to drop columns
        if st.checkbox('Drop Columns'):
            columns_to_drop = st.multiselect('Select columns to drop', df.columns)
            if st.button('Drop Columns'):
                df = df.drop(columns=columns_to_drop)
                st.write(df.head())
                st.success('Columns dropped successfully!')






        # Perform Exploratory Data Analysis
        perform_eda(df)

        # Handle missing values
        df = handle_missing_values(df)

        # Handle categorical data encoding
        df = handle_categorical_encoding(df)

        # Select target variable
        target_var = st.selectbox('Select target variable', df.columns)
        if target_var:
            unique_values = df[target_var].nunique()
            task_type = 'classification' if unique_values < 10 else 'regression'
            st.write(f'Task Type: {task_type.capitalize()}')

            if st.button('Setup and Train Models'):
                if task_type == 'regression':
                    # Setup data for training (regression)
                    reg = reg_setup(data=df, target=target_var, train_size=0.8, fold=3)

                    # Train and compare multiple regression models
                    st.info('Training and comparing models...')
                    reg_compare_models()

                    # Get performance metrics table
                    metrics_table = reg_pull()
                    st.subheader('Performance Metrics:')
                    st.write(metrics_table)

                    # Get the name of the best model
                    best_model_name = reg_pull().index[0]

                    # Create and train the best model
                    best_model = reg_create_model(best_model_name)
                    st.info(f'Training best model ({best_model_name})...')
                    model_trained = reg_predict_model(best_model)
                    st.success('Best model trained!')

                    # Tune the best model
                    st.info('Tuning the best model...')
                    # tuned_model = reg_tune_model(best_model, n_iter=10)
                    custom_grid = {
                        'n_estimators': [10, 50, 100],
                        'max_depth': [3, 5, 7]
                    }
                    tuned_model = reg_tune_model(best_model, custom_grid=custom_grid, n_iter=5)

                    # Show best-tuned model results
                    st.subheader('Best-Tuned Model Results:')
                    st.write(tuned_model)
                    st.write(reg_pull())
                    st.subheader(' Model  Expectations Results:')
                    st.write(model_trained)

                elif task_type == 'classification':
                    # Setup data for training (classification)
                    cls = cls_setup(data=df, target=target_var, train_size=0.8, fold=3)

                    # Train and compare multiple classification models
                    st.info('Training and comparing models...')
                    cls_compare_models()

                    # Get performance metrics table
                    metrics_table = cls_pull()
                    st.subheader('Performance Metrics:')
                    st.write(metrics_table)

                    # Get the name of the best model
                    best_model_name = cls_pull().index[0]

                    # Create and train the best model
                    best_model = cls_create_model(best_model_name)
                    st.info(f'Training best model ({best_model_name})...')
                    model_trained = cls_predict_model(best_model)
                    st.success('Best model trained!')

                    # Tune the best model
                    st.info('Tuning the best model...')

                    # tuned_model = cls_tune_model(best_model,n_iter=10)

                    custom_grid = {
                    'n_estimators': [10, 50, 100],
                    'max_depth': [3, 5, 7]
                    }
                    tuned_model = cls_tune_model(best_model, custom_grid=custom_grid, n_iter=5)

                    # Show best-tuned model results
                    st.subheader('Best-Tuned Model Results:')
                    st.write(tuned_model)
                    st.write(cls_pull())
                    st.subheader(' Model  Expectations Results:')
                    st.write(model_trained)

                # Plot the performance of the best-tuned model
                st.subheader('Done')

if __name__ == '__main__':
    main()










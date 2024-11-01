# Import necessary libraries
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb
import holidays
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


scaler = MinMaxScaler()
# Features Functions
def create_datetime_features(df):
    """
    Creates time series features from datetime index
    """
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.isocalendar().week
    df['date'] = pd.to_datetime(df.index.date)
    return df

def show_forecast_page():
    st.title("PJM Dominion Energy Consumption Forecasting App ")

    # Load the trained XGBoost model
    model_path = 'saved_models/edf_xgb_best_model_gs_RTall.joblib'
    model = joblib.load(model_path)

    # Load the fitted MinMaxScaler (ensure you save the scaler after fitting during training)
    scaler_path = 'saved_models/minmax_scaler.joblib'  # Adjust path if necessary
    scaler = joblib.load(scaler_path)

    # Input: Get future date for prediction
    st.write("#### Please select a forecasting period:")
    start_date = st.date_input("Select a start date", datetime.today())
    end_date = st.date_input("Select an end date", datetime.today())

    # Generate a date range with hourly intervals
    future = pd.date_range(start=start_date, end=end_date, freq='1H')
    future_df = pd.DataFrame(index=future)

    # Apply datetime features to the future_df
    future_df = create_datetime_features(future_df)

    # Drop the columns that were removed in your Jupyter Notebook
    train_drop_cols = ['date']  # Adjust this if you have more columns to drop
    X_future = future_df.drop(columns=train_drop_cols)

    # Scale the features of the future dataframe using the loaded scaler
    scaled_future_df = scaler.transform(X_future)

    # Predict using the loaded model
    future_predictions_scaled = model.predict(scaled_future_df)

    # Inverse transform to get the original scale
    #future_predictions = scaler.inverse_transform(future_predictions_scaled.reshape(-1, 1))
    # Use the predicted values directly
    future_predictions = future_predictions_scaled

    # Create a DataFrame with future dates and predicted values
    forecast_df = pd.DataFrame({
        'date': future_df.index,
        'Predicted Consumption (MW)': future_predictions.flatten()
    })

    # Plot the predictions
    #st.line_chart(forecast_df.set_index('date'))


    # Plot the predictions using Matplotlib
    plt.figure(figsize=(10, 5))
    plt.plot(forecast_df['date'], forecast_df['Predicted Consumption (MW)'], color='red', label='Predicted Consumption (MW)')
    plt.title('Energy Consumption Forecast')
    plt.xlabel('Date')
    plt.ylabel('Predicted Consumption (MW)')
    plt.xticks(rotation=45)
    plt.grid()
    plt.legend()
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(plt)

    # Export predictions as CSV
    if st.button("Export forecast to CSV"):
        csv = forecast_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='energy_forecast.csv',
            mime='text/csv'
        )
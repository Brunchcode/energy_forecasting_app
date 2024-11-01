# Import necessary libraries
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from forecast import create_datetime_features

# Function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Function to load actuals and forecasted data for the selected date range
def load_actuals_and_forecast(start_date, end_date):
    # Convert datetime.date to pd.Timestamp for consistency
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    # Load and clean the CSV file, skipping the first row which contains headers
    actuals_path = 'saved_models/actuals.csv'  # Path to the actual data

    # Load the actuals data, skipping the first row
    actuals_df = pd.read_csv(actuals_path, 
                             skiprows=1, 
                             header=None, 
                             names=['datetime', 'actual_consumption'], 
                             parse_dates=['datetime'], 
                             dayfirst=True)

    # Set 'datetime' as the index
    actuals_df.set_index('datetime', inplace=True)

    # Ensure the actuals datetime index is in pd.Timestamp format
    actuals_df.index = pd.to_datetime(actuals_df.index)

    # Filter actuals for the selected date range (now hourly)
    actuals_df = actuals_df.loc[start_date:end_date]

    # Load the trained XGBoost model
    model_path = 'saved_models/edf_xgb_best_model_gs_RTall.joblib'
    model = joblib.load(model_path)

    # Load the fitted MinMaxScaler
    scaler_path = 'saved_models/minmax_scaler.joblib'
    scaler = joblib.load(scaler_path)

    # Generate a date range with hourly intervals
    future = pd.date_range(start=start_date, end=end_date, freq='1H')
    future_df = pd.DataFrame(index=future)

    # Apply datetime features to the future_df
    future_df = create_datetime_features(future_df)

    # Drop columns used only for training, like 'date'
    train_drop_cols = ['date']
    X_future = future_df.drop(columns=train_drop_cols)

    # Scale the features using the loaded scaler
    scaled_future_df = scaler.transform(X_future)

    # Predict using the loaded model
    future_predictions = model.predict(scaled_future_df)

    # Create a DataFrame with future dates and predicted values
    forecast_df = pd.DataFrame({
        'date': future_df.index,
        'Forecasted Consumption (MW)': future_predictions.flatten()
    }).set_index('date')

    return actuals_df, forecast_df


# Function to show evaluation page
def show_evaluation_page():
    st.title("Evaluate Energy Consumption Forecast")

    # User input to select evaluation dates
    st.write("#### Please select an evaluation period:")
    start_date = st.date_input("Select a start date", value=pd.to_datetime('2024-04-01'))
    end_date = st.date_input("Select an end date", value=pd.to_datetime('2024-08-31'))

    if start_date > end_date:
        st.error("Start date must be before end date.")
        return

    # Load actuals and forecasted data for the selected range
    actuals_df, forecast_df = load_actuals_and_forecast(start_date, end_date)

    # Combine actual and forecast data into a single DataFrame
    evaluation_df = pd.concat([actuals_df, forecast_df], axis=1).dropna()

    if evaluation_df.empty:
        st.error("No data available for the selected date range.")
        return

    # Plot 1: Actual vs Forecast Line Plot
    st.subheader("Actual vs Forecast: Energy Consumption Over Time")
    plt.figure(figsize=(10, 5))
    plt.plot(evaluation_df.index, evaluation_df['actual_consumption'], label='Actual Consumption (MW)', color='blue')
    plt.plot(evaluation_df.index, evaluation_df['Forecasted Consumption (MW)'], label='Forecasted Consumption (MW)', color='red')
    plt.title('Actual vs Forecasted Energy Consumption (MW)')
    plt.xlabel('Date')
    plt.ylabel('Consumption (MW)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

    # Plot 2: Scatter Plot (Actual vs Forecast)
    st.subheader("Scatter Plot: Actual vs Forecasted Consumption")
    plt.figure(figsize=(10, 6))
    plt.scatter(evaluation_df['actual_consumption'], evaluation_df['Forecasted Consumption (MW)'], color='blue', alpha=0.5)
    plt.plot([evaluation_df['actual_consumption'].min(), evaluation_df['actual_consumption'].max()],
             [evaluation_df['actual_consumption'].min(), evaluation_df['actual_consumption'].max()], 
             color='red', linestyle='--')
    plt.title('Scatter Plot: Actual vs Forecast')
    plt.xlabel('Actual Consumption (MW)')
    plt.ylabel('Forecasted Consumption (MW)')
    plt.tight_layout()
    st.pyplot(plt)

    # Calculate evaluation metrics
    mae = mean_absolute_error(evaluation_df['actual_consumption'], evaluation_df['Forecasted Consumption (MW)'])
    rmse = np.sqrt(mean_squared_error(evaluation_df['actual_consumption'], evaluation_df['Forecasted Consumption (MW)']))
    mape = mean_absolute_percentage_error(evaluation_df['actual_consumption'], evaluation_df['Forecasted Consumption (MW)'])

    # Display metrics in a table
    st.subheader("Evaluation Metrics")
    metrics_df = pd.DataFrame({
        'Metric': ['MAE', 'RMSE', 'MAPE'],
        'Value': [mae, rmse, mape]
    })
    st.table(metrics_df)

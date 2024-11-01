import streamlit as st
from forecast import show_forecast_page
from evaluate import show_evaluation_page

# Page selection
page = st.sidebar.selectbox("Evaluate or Forecast", ("Forecast", "Evaluate"))

if page == "Forecast":
    show_forecast_page()
elif page == "Evaluate":
    show_evaluation_page()
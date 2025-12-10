import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import gdown  # pip install gdown

# -----------------------------
# Download model if not present
# -----------------------------
MODEL_PATH = "crime_rate_model.pkl"
MODEL_URL = "YOUR_GOOGLE_DRIVE_FILE_ID_OR_LINK"  # replace with your model link

if not os.path.exists(MODEL_PATH):
    st.info("Downloading trained model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    st.success("Model downloaded!")

# -----------------------------
# Load trained model
# -----------------------------
saved = joblib.load(MODEL_PATH)
model = saved["model"]
features = saved["features"]
category_columns = saved["category_columns"]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🔮 Multi-Category Crime Rate Prediction - Next 60 Days")
st.write("Select one or more crime categories to predict the next 60 days of crime counts starting from tomorrow.")

# Multi-category selection
category_names = [c.replace("Crime_Category_", "") for c in category_columns]
crime_choices = st.multiselect("Select Crime Categories", category_names, default=[category_names[0]])

if st.button("Predict"):
    if not crime_choices:
        st.warning("Please select at least one category.")
    else:
        # Generate future 60 days starting tomorrow
        start_date = datetime.today() + timedelta(days=1)
        future_dates = pd.date_range(start=start_date, periods=60)

        # Create empty dataframe
        future_df = pd.DataFrame({
            "Date": future_dates,
            "Day": future_dates.day,
            "Month": future_dates.month,
            "Year": future_dates.year,
            "DayOfWeek": future_dates.weekday
        })

        # Create figure for plotting
        fig, ax = plt.subplots(figsize=(12, 5))

        # Dataframe to collect all predictions
        combined_results = pd.DataFrame({"Date": future_dates})

        for crime in crime_choices:
            # One-hot encode all categories
            for col in category_columns:
                future_df[col] = 0
            future_df[f"Crime_Category_{crime}"] = 1

            # Prepare features
            X_future = future_df[features]

            # Predict
            preds = model.predict(X_future)

            # Confidence intervals ±10%
            lower = preds * 0.9
            upper = preds * 1.1

            # Plot line and CI
            ax.plot(future_dates, preds, label=f"{crime} Prediction")
            ax.fill_between(future_dates, lower, upper, alpha=0.2)

            # Add to combined results
            combined_results[f"{crime}_Prediction"] = preds
            combined_results[f"{crime}_Lower_CI"] = lower
            combined_results[f"{crime}_Upper_CI"] = upper

        # Customize plot
        ax.set_xlabel("Date")
        ax.set_ylabel("Crime Count")
        ax.set_title("Next 60 Days Crime Predictions")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Show combined table
        st.subheader("Predicted Data Table")
        st.dataframe(combined_results)

        # CSV download
        csv = combined_results.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions CSV", csv, "next_60_days_prediction.csv")

        st.success("Prediction completed! Dates start from tomorrow.")




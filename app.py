import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import requests

# -----------------------------
# CONFIG: MODEL DOWNLOAD
# -----------------------------
MODEL_PATH = "crime_rate_model.pkl"

# Choose one of the two options:

# 1️⃣ GitHub Release link (recommended if model <= 100MB)
MODEL_URL = "https://github.com/YOUR_USERNAME/YOUR_REPO/releases/download/v1.0/crime_rate_model.pkl"

# 2️⃣ OR Google Drive link
# MODEL_URL = "https://drive.google.com/uc?id=YOUR_FILE_ID"

# Function to download model
def download_model(url, path):
    st.info("Downloading trained model...")
    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success("Model downloaded successfully!")
    except Exception as e:
        st.error(f"Failed to download model: {e}")
        st.stop()

# Download model if not present
if not os.path.exists(MODEL_PATH):
    download_model(MODEL_URL, MODEL_PATH)

# -----------------------------
# LOAD MODEL
# -----------------------------
saved = joblib.load(MODEL_PATH)
model = saved["model"]
features = saved["features"]
category_columns = saved["category_columns"]

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🔮 Multi-Category Crime Rate Prediction - Next 60 Days")
st.write("Select one or more crime categories to predict the next 60 days of crime counts starting from tomorrow.")

category_names = [c.replace("Crime_Category_", "") for c in category_columns]
crime_choices = st.multiselect("Select Crime Categories", category_names, default=[category_names[0]])

if st.button("Predict"):
    if not crime_choices:
        st.warning("Please select at least one category.")
    else:
        start_date = datetime.today() + timedelta(days=1)
        future_dates = pd.date_range(start=start_date, periods=60)

        future_df = pd.DataFrame({
            "Date": future_dates,
            "Day": future_dates.day,
            "Month": future_dates.month,
            "Year": future_dates.year,
            "DayOfWeek": future_dates.weekday
        })

        fig, ax = plt.subplots(figsize=(12,5))
        combined_results = pd.DataFrame({"Date": future_dates})

        for crime in crime_choices:
            for col in category_columns:
                future_df[col] = 0
            future_df[f"Crime_Category_{crime}"] = 1

            X_future = future_df[features]
            preds = model.predict(X_future)

            # Confidence intervals ±10%
            lower = preds * 0.9
            upper = preds * 1.1

            # Plot
            ax.plot(future_dates, preds, label=f"{crime} Prediction")
            ax.fill_between(future_dates, lower, upper, alpha=0.2)

            # Add to results
            combined_results[f"{crime}_Prediction"] = preds
            combined_results[f"{crime}_Lower_CI"] = lower
            combined_results[f"{crime}_Upper_CI"] = upper

        ax.set_xlabel("Date")
        ax.set_ylabel("Crime Count")
        ax.set_title("Next 60 Days Crime Predictions")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.subheader("Predicted Data Table")
        st.dataframe(combined_results)

        csv = combined_results.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions CSV", csv, "next_60_days_prediction.csv")

        st.success("Prediction completed! Dates start from tomorrow.")







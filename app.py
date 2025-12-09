import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

st.title("Chicago Crime Forecast (Next 60 Days) — XGBoost Version")
st.write("Fast loading + No Prophet + XGBoost time-series model")


# ---------------------------------------------------------
# FAST CHUNK LOADING (ONLY DATE COLUMN)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    chunks = pd.read_csv(
        "Chicago_Crimes_2012_to_2017.csv",
        usecols=["Date"],
        chunksize=100000
    )
    df = pd.concat(chunks)

    # Clean / convert date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.dropna(inplace=True)

    # Daily crime count
    daily = df.groupby(df["Date"].dt.date).size().reset_index()
    daily.columns = ["date", "count"]
    daily["date"] = pd.to_datetime(daily["date"])

    # Use last 2 years → extremely fast + more accurate
    recent = daily[daily["date"] > daily["date"].max() - pd.Timedelta(days=730)]

    return recent


data = load_data()


# ---------------------------------------------------------
# FEATURE ENGINEERING FOR XGBOOST
# ---------------------------------------------------------
def create_features(df):
    df = df.copy()
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    return df


data = create_features(data)

st.write("### Last 2 Years of Crime Data")
st.line_chart(data.set_index("date")["count"])


# ---------------------------------------------------------
# TRAIN XGBOOST
# ---------------------------------------------------------
st.write("### Training Fast XGBoost Model...")

feature_cols = ["day", "month", "year", "day_of_week", "week_of_year"]

X = data[feature_cols]
y = data["count"]

model = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    objective="reg:squarederror",
    tree_method="hist"  # super fast on CPU
)

model.fit(X, y)


# ---------------------------------------------------------
# PREDICT NEXT 60 DAYS
# ---------------------------------------------------------
st.write("### Predicting Next 60 Days...")

last_date = data["date"].max()
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=60)

future_df = pd.DataFrame({"date": future_dates})
future_df = create_features(future_df)

future_X = future_df[feature_cols]
future_pred = model.predict(future_X)

future_df["forecast"] = future_pred


# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------
st.write("### Next 60 Days Forecast")
st.line_chart(future_df.set_index("date")["forecast"])

st.success("Forecast Complete! ⚡ (XGBoost model is extremely fast)")  


import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler

in_path = "AirQualityData.csv"
out_clean = "AirQuality_clean.csv"
out_scaled = "AirQuality_scaled.csv"

# Loading my file
df = pd.read_csv("D:\AICTE\Project\AirQualityData.csv")

# Build a proper date and time column
df["Datetime"] = pd.to_datetime(
    df["Date"].astype(str) + " " + df["Time"].astype(str),
    errors="coerce", dayfirst=True
)
df.drop(columns=[c for c in ["Date", "Time"] if c in df.columns], inplace=True, errors="ignore")
df = df[["Datetime"] + [c for c in df.columns if c != "Datetime"]]

# Operating the columns with only having a numerical values
num_cols = df.select_dtypes(include=[np.number]).columns

# Replace invalid values with NaN
df[num_cols] = df[num_cols].mask(df[num_cols] < 0)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Drop constant (no-variance) columns 
const_cols = [c for c in num_cols if df[c].nunique(dropna=True) <= 1]
if const_cols:
    df.drop(columns=const_cols, inplace=True)
    num_cols = df.select_dtypes(include=[np.number]).columns  # refresh

# Impute missing numeric values
imputer = KNNImputer(n_neighbors=5, weights="distance")
df[num_cols] = imputer.fit_transform(df[num_cols])

# Save Clean Data
df.to_csv(out_clean, index=False)

# Scale numeric features for modeling
scaler = RobustScaler()
df_scaled = df.copy()
df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols])

# Saving the file
df_scaled.to_csv(out_scaled, index=False)

print("Saved:", out_clean, "and", out_scaled)

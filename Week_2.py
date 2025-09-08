
#         ------------ Week 2-----------
# Implementataion of the ML model

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load the CLEANED data (use scaled if you want normalized inputs)
df = pd.read_csv("AirQuality_clean.csv")

# Target = PM2.5, Features = all other numeric columns
X = df.drop(columns=["PM2.5", "Datetime"])   # features
y = df["PM2.5"]                              # target

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Train & Evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"📌 {name} Results:")
    print(f"   MAE  = {mae:.3f}")
    print(f"   RMSE = {rmse:.3f}")
    print(f"   R²   = {r2:.3f}")
    print("-"*40)


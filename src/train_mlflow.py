import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

# =========================
# Load Dataset
# =========================

df = pd.read_csv("../data/smart_grid_dataset.csv")

# =========================
# Feature Engineering
# =========================

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

df["hour"] = df["Timestamp"].dt.hour
df["day"] = df["Timestamp"].dt.day
df["month"] = df["Timestamp"].dt.month

df = df.drop("Timestamp", axis=1)

# =========================
# Define Features & Target
# =========================

target = "Grid Supply (kW)"

X = df.drop(columns=[target])
y = df[target]

# =========================
# Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# MLflow Experiment
# =========================

mlflow.set_experiment("Smart_Grid_Load_Prediction")

models = {
    "LinearRegression": LinearRegression(),

    "RandomForest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        objective="reg:squarederror"
    )
}

# =========================
# Train and Log Models
# =========================

for model_name, model in models.items():

    with mlflow.start_run(run_name=model_name):

        # Train
        model.fit(X_train, y_train)

        # Predict
        preds = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2 = r2_score(y_test, preds)

        # Log Params
        mlflow.log_param("model_name", model_name)

        # Log Metrics
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        # Log Model
        if model_name == "XGBoost":
            mlflow.xgboost.log_model(
                xgb_model=model,
                name="model"
            )
        else:
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model"
            )

        # Print Results
        print("\n======================")
        print(f"Model: {model_name}")
        print(f"MAE  : {mae:.6f}")
        print(f"RMSE : {rmse:.6f}")
        print(f"R2   : {r2:.6f}")
        print("======================")

print("\nAll models logged successfully to MLflow!")
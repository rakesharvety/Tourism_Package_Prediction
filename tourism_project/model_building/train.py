"""Model training with hyperparameter tuning and MLflow experiment tracking.

Loads the train/test splits produced by the data-prep job, tunes an XGBoost
classifier with GridSearchCV, logs every trial to MLflow, evaluates the best
model on train and test data, and saves it so the workflow can commit it to the
repository (tourism_project/deployment/).
"""
import contextlib
import io
import os
import urllib.request

import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_PATH = "tourism_project/deployment/best_tourism_model_v1.joblib"

# --- MLflow setup -----------------------------------------------------------
# The workflow starts an MLflow server on port 5000. If it is not reachable
# (for example when running this script on a laptop), fall back to a local
# ./mlruns folder so training never fails just because tracking is unavailable.
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
try:
    # MLflow servers answer "OK" on /health; anything else means not our server
    urllib.request.urlopen(f"{TRACKING_URI}/health", timeout=5)
except Exception:
    print(f"MLflow server not reachable at {TRACKING_URI}; using local ./mlruns")
    TRACKING_URI = "file:./mlruns"
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("tourism-package-prediction")

# --- Load data (artifact from the previous job) -----------------------------
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

# --- Features ---------------------------------------------------------------
numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting",
    "MonthlyIncome",
]
categorical_features = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]

# --- Preprocessing: impute, scale numerics, one-hot encode categoricals -----
preprocessor = make_column_transformer(
    (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
     numeric_features),
    (make_pipeline(SimpleImputer(strategy="most_frequent"),
                   OneHotEncoder(handle_unknown="ignore")),
     categorical_features),
)

# Class imbalance: only ~19% of customers buy the package
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight, random_state=42, eval_metric="logloss"
)
model_pipeline = make_pipeline(preprocessor, xgb_model)

# --- Hyperparameter grid ----------------------------------------------------
param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__colsample_bytree": [0.6, 0.8],
    "xgbclassifier__reg_lambda": [1, 5],
}

CLASSIFICATION_THRESHOLD = 0.5

with mlflow.start_run(run_name="xgboost-grid-search"):
    # F1 balances catching buyers (recall) against wasted outreach (precision)
    grid_search = GridSearchCV(
        model_pipeline, param_grid, cv=3, scoring="f1", n_jobs=-1
    )
    grid_search.fit(Xtrain, ytrain)

    # Log every parameter combination as its own nested run
    # (MLflow prints a "View run" line per run; silence it to keep the log readable)
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with mlflow.start_run(nested=True, run_name=f"trial-{i}"):
                mlflow.log_params(results["params"][i])
                mlflow.log_metric("mean_cv_f1", results["mean_test_score"][i])
                mlflow.log_metric("std_cv_f1", results["std_test_score"][i])
    print(f"Logged {len(results['params'])} tuning trials to MLflow ({TRACKING_URI})")

    # Log the best parameters on the parent run
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_f1", grid_search.best_score_)
    print("Best parameters:", grid_search.best_params_)
    print(f"Best CV F1: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_

    # --- Evaluation ---------------------------------------------------------
    def evaluate(name, X, y):
        proba = best_model.predict_proba(X)[:, 1]
        pred = (proba >= CLASSIFICATION_THRESHOLD).astype(int)
        print(f"\n{name} classification report")
        print(classification_report(y, pred, target_names=["No purchase", "Purchase"]))
        return {
            f"{name}_accuracy": accuracy_score(y, pred),
            f"{name}_precision": precision_score(y, pred),
            f"{name}_recall": recall_score(y, pred),
            f"{name}_f1": f1_score(y, pred),
            f"{name}_roc_auc": float(roc_auc_score(y, proba)),
        }

    metrics = {}
    metrics.update(evaluate("train", Xtrain, ytrain))
    metrics.update(evaluate("test", Xtest, ytest))
    mlflow.log_metrics(metrics)
    print("\nMetrics:", {k: round(v, 4) for k, v in metrics.items()})

    # --- Save the best model so the workflow can commit it ------------------
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"Best model saved to {MODEL_PATH}")

    # Also attach the model to the MLflow run (non-critical)
    try:
        mlflow.sklearn.log_model(best_model, name="model")
    except Exception as e:  # artifact logging must never break the pipeline
        print("Skipped MLflow model artifact logging:", e)

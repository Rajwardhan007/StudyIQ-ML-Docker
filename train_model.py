
import os
import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load dataset
DATA_PATH = "data/student-mat.csv"

data = pd.read_csv(DATA_PATH, sep=";")

print("Dataset loaded successfully!")
print("Records:", len(data))
print("Columns:", len(data.columns))


# 2. Separate input features and target
X = data.drop(columns=["G3"])
y = data["G3"]


# 3. Identify numerical and categorical columns
numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


# 4. Preprocessing
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])


# 5. Build ML pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])


# 6. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# 7. Train model
print("\nTraining model...")
model.fit(X_train, y_train)


# 8. Evaluate model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\nMODEL EVALUATION")
print("-------------------------")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")


# 9. Save trained model
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/student_model.pkl")

# Save default values for dashboard inputs
defaults = {}

for column in X.columns:
    if X[column].dtype == "object":
        defaults[column] = str(X[column].mode()[0])
    else:
        defaults[column] = float(X[column].median())

with open("models/default_student.json", "w") as file:
    json.dump(defaults, file, indent=4)

print("Default student profile saved!")

metrics = {
    "MAE": round(float(mae), 3),
    "RMSE": round(float(rmse), 3),
    "R2": round(float(r2), 3)
}

with open("models/metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("\nModel saved successfully!")
print("Location: models/student_model.pkl")
print("Evaluation saved: models/metrics.json")

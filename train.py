import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("student_data.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print("Columns in dataset:")
print(df.columns)

# ❗ REMOVE rows where target is missing
df = df.dropna(subset=["finalgrade"])

# Target
y = df["finalgrade"]

# Remove useless columns
df = df.drop(["studentid", "name"], axis=1)

# Features
X = df.drop("finalgrade", axis=1)

# Column types
categorical_cols = [
    "gender",
    "extracurricularactivities",
    "parentalsupport"
]

numerical_cols = [
    "attendancerate",
    "studyhoursperweek",
    "previousgrade",
    "study_hours",
    "attendance_(%)",
    "online_classes_taken"
]

# Pipelines for missing values
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean"))
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", num_pipeline, numerical_cols),
    ("cat", cat_pipeline, categorical_cols)
])

# Full pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

print("\n✅ Model trained successfully (NaN fixed)!")
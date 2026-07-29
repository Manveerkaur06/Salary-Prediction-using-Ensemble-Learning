from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder



BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Salary_Data.csv"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)


#load the dataset 
raw_data = pd.read_csv(DATA_PATH)
raw_data.columns = [
    column.strip().lstrip("\ufeff")
    for column in raw_data.columns
]

required_columns = {
    "Age",
    "Gender",
    "Education Level",
    "Job Title",
    "Years of Experience",
    "Salary",
}

if not required_columns.issubset(raw_data.columns):
    raise ValueError(
        "Dataset must contain Age, Gender, Education Level, Job Title, "
        "Years of Experience and Salary columns."
    )


# clean dataset
raw_row_count = len(raw_data)
raw_duplicate_count = int(raw_data.duplicated().sum())
raw_missing_rows = int(raw_data.isnull().any(axis=1).sum())

data = raw_data[list(required_columns)].dropna().copy()


education_replacements = {
    "Bachelor's": "Bachelor's Degree",
    "Master's": "Master's Degree",
    "phD": "PhD",
}
data["Education Level"] = data["Education Level"].replace(
    education_replacements
)

categorical_columns = ["Gender", "Education Level", "Job Title"]
numerical_columns = ["Age", "Years of Experience"]

for column in categorical_columns:
    data[column] = data[column].astype(str).str.strip()


data = data[
    (data["Age"] >= 18)
    & (data["Years of Experience"] >= 0)
    & (data["Salary"] >= 10000)
]

data = data.drop_duplicates().reset_index(drop=True)
clean_row_count = len(data)



feature_columns = [
    "Age",
    "Gender",
    "Education Level",
    "Job Title",
    "Years of Experience",
]

X = data[feature_columns]
y = data["Salary"]


# split data into train test 
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)



def create_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("numerical", "passthrough", numerical_columns),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_columns,
            ),
        ]
    )


# random forest (bagging model)
random_forest_pipeline = Pipeline(
    steps=[
        ("preprocessor", create_preprocessor()),
        (
            "model",
            RandomForestRegressor(
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

random_forest_grid = {
    "model__n_estimators": [150, 250],
    "model__max_depth": [None, 15],
    "model__min_samples_leaf": [1, 2],
}

random_forest_search = GridSearchCV(
    estimator=random_forest_pipeline,
    param_grid=random_forest_grid,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
)
random_forest_search.fit(X_train, y_train)
random_forest_model = random_forest_search.best_estimator_


#gradient boosting
gradient_boosting_pipeline = Pipeline(
    steps=[
        ("preprocessor", create_preprocessor()),
        (
            "model",
            GradientBoostingRegressor(
                random_state=42,
            ),
        ),
    ]
)

gradient_boosting_grid = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.05, 0.10],
    "model__max_depth": [2, 3],
}

gradient_boosting_search = GridSearchCV(
    estimator=gradient_boosting_pipeline,
    param_grid=gradient_boosting_grid,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
)
gradient_boosting_search.fit(X_train, y_train)
gradient_boosting_model = gradient_boosting_search.best_estimator_


# evaluate both models
def calculate_metrics(model_name, model, X_value, y_value):
    predictions = model.predict(X_value)
    mse = mean_squared_error(y_value, predictions)

    return {
        "Model": model_name,
        "MAE": mean_absolute_error(y_value, predictions),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2_Score": r2_score(y_value, predictions),
        "Predictions": predictions,
    }


forest_result = calculate_metrics(
    "Random Forest Regressor",
    random_forest_model,
    X_test,
    y_test,
)

boosting_result = calculate_metrics(
    "Gradient Boosting Regressor",
    gradient_boosting_model,
    X_test,
    y_test,
)

results = [forest_result, boosting_result]

comparison = pd.DataFrame(
    [
        {
            "Model": result["Model"],
            "MAE": round(result["MAE"], 2),
            "MSE": round(result["MSE"], 2),
            "RMSE": round(result["RMSE"], 2),
            "R2_Score": round(result["R2_Score"], 4),
        }
        for result in results
    ]
)
comparison.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)


# save clean data
data.to_csv(OUTPUTS_DIR / "cleaned_salary_data.csv", index=False)

test_predictions = X_test.copy()
test_predictions["Actual_Salary"] = y_test
test_predictions["Random_Forest_Prediction"] = forest_result["Predictions"]
test_predictions["Gradient_Boosting_Prediction"] = boosting_result["Predictions"]
test_predictions = test_predictions.sort_values(
    ["Years of Experience", "Age"]
)
test_predictions.to_csv(
    OUTPUTS_DIR / "test_predictions.csv",
    index=False,
)


# create graphs
plt.figure(figsize=(8, 5))
plt.scatter(
    data["Years of Experience"],
    data["Salary"],
    color="#2563eb",
    alpha=0.45,
)
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Experience and Salary in the Clean Dataset")
plt.grid(alpha=0.20)
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "salary_dataset.png", dpi=180)
plt.close()


plt.figure(figsize=(8, 5))
plt.hist(
    data["Salary"],
    bins=20,
    color="#2563eb",
    edgecolor="white",
)
plt.xlabel("Salary")
plt.ylabel("Number of Records")
plt.title("Salary Distribution")
plt.grid(axis="y", alpha=0.20)
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "salary_distribution.png", dpi=180)
plt.close()


plt.figure(figsize=(8, 5))
plt.bar(
    comparison["Model"],
    comparison["R2_Score"],
    color=["#2563eb", "#16a34a"],
)
plt.ylim(0, 1)
plt.ylabel("R² Score")
plt.title("Model Performance Comparison")
for index, score in enumerate(comparison["R2_Score"]):
    plt.text(index, score + 0.02, f"{score:.3f}", ha="center")
plt.xticks(rotation=8)
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "model_comparison.png", dpi=180)
plt.close()


fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

axes[0].scatter(
    y_test,
    forest_result["Predictions"],
    color="#2563eb",
    alpha=0.65,
)
axes[0].set_title("Random Forest")
axes[0].set_xlabel("Actual Salary")
axes[0].set_ylabel("Predicted Salary")
axes[0].grid(alpha=0.20)

axes[1].scatter(
    y_test,
    boosting_result["Predictions"],
    color="#16a34a",
    alpha=0.65,
)
axes[1].set_title("Gradient Boosting")
axes[1].set_xlabel("Actual Salary")
axes[1].grid(alpha=0.20)

minimum_value = min(
    y_test.min(),
    *forest_result["Predictions"],
    *boosting_result["Predictions"],
)
maximum_value = max(
    y_test.max(),
    *forest_result["Predictions"],
    *boosting_result["Predictions"],
)

for axis in axes:
    axis.plot(
        [minimum_value, maximum_value],
        [minimum_value, maximum_value],
        color="#dc2626",
        linestyle="--",
        label="Perfect prediction",
    )
    axis.legend()

fig.suptitle("Actual Salary vs Predicted Salary")
fig.tight_layout()
fig.savefig(OUTPUTS_DIR / "actual_vs_predicted.png", dpi=180)
plt.close(fig)


#retrain on clean data
final_random_forest_model = clone(random_forest_model)
final_random_forest_model.fit(X, y)

final_gradient_boosting_model = clone(gradient_boosting_model)
final_gradient_boosting_model.fit(X, y)



forest_preprocessor = final_random_forest_model.named_steps["preprocessor"]
forest_regressor = final_random_forest_model.named_steps["model"]

feature_names = forest_preprocessor.get_feature_names_out()
feature_importances = pd.Series(
    forest_regressor.feature_importances_,
    index=feature_names,
).sort_values(ascending=False)

top_features = feature_importances.head(15).sort_values()

plt.figure(figsize=(9, 6))
top_features.plot(kind="barh", color="#2563eb")
plt.xlabel("Importance")
plt.title("Top 15 Random Forest Feature Importances")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "feature_importance.png", dpi=180)
plt.close()

feature_importances.rename("Importance").to_csv(
    OUTPUTS_DIR / "feature_importance.csv",
)



forest_model_path = MODELS_DIR / "random_forest_model.joblib"
boosting_model_path = MODELS_DIR / "gradient_boosting_model.joblib"

joblib.dump(final_random_forest_model, forest_model_path)
joblib.dump(final_gradient_boosting_model, boosting_model_path)



def remove_pipeline_prefix(parameters):
    return {
        key.replace("model__", ""): value
        for key, value in parameters.items()
    }


best_model_name = comparison.sort_values(
    "R2_Score",
    ascending=False,
).iloc[0]["Model"]

model_information = {
    "raw_dataset_rows": int(raw_row_count),
    "raw_duplicate_rows": int(raw_duplicate_count),
    "raw_rows_with_missing_values": int(raw_missing_rows),
    "clean_dataset_rows": int(clean_row_count),
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
    "feature_count": int(len(feature_columns)),
    "feature_columns": feature_columns,
    "test_size": 0.20,
    "random_state": 42,
    "best_model_on_test_data": best_model_name,
    "random_forest_best_parameters": remove_pipeline_prefix(
        random_forest_search.best_params_
    ),
    "gradient_boosting_best_parameters": remove_pipeline_prefix(
        gradient_boosting_search.best_params_
    ),
    "age_min": float(data["Age"].min()),
    "age_max": float(data["Age"].max()),
    "experience_min": float(data["Years of Experience"].min()),
    "experience_max": float(data["Years of Experience"].max()),
    "gender_options": sorted(data["Gender"].unique().tolist()),
    "education_options": sorted(
        data["Education Level"].unique().tolist()
    ),
    "job_title_options": sorted(data["Job Title"].unique().tolist()),
}

with open(
    OUTPUTS_DIR / "model_information.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(model_information, file, indent=4)


print("\nDataset summary:")
print(f"Raw rows: {raw_row_count}")
print(f"Clean unique rows: {clean_row_count}")
print(f"Input features: {len(feature_columns)}")

print("\nModel comparison on unseen test data:\n")
print(comparison.to_string(index=False))

print("\nBest Random Forest parameters:")
print(remove_pipeline_prefix(random_forest_search.best_params_))

print("\nBest Gradient Boosting parameters:")
print(remove_pipeline_prefix(gradient_boosting_search.best_params_))

print("\nSaved model pipelines:")
print(forest_model_path)
print(boosting_model_path)

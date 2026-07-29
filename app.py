from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"


st.set_page_config(
    page_title="Salary Prediction",
    page_icon="💼",
    layout="centered",
)


@st.cache_resource
def load_saved_model(model_path):
    return joblib.load(model_path)


@st.cache_data
def load_model_comparison():
    return pd.read_csv(OUTPUTS_DIR / "model_comparison.csv")


@st.cache_data
def load_model_information():
    with open(
        OUTPUTS_DIR / "model_information.json",
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


model_information = load_model_information()


st.title("Salary Prediction System")
st.write(
    "Enter the employee details and select an ensemble model "
    "to estimate the salary."
)


model_options = {
    "Random Forest Regressor (Bagging Model)": (
        MODELS_DIR / "random_forest_model.joblib"
    ),
    "Gradient Boosting Regressor (Boosting Model)": (
        MODELS_DIR / "gradient_boosting_model.joblib"
    ),
}

selected_model_name = st.selectbox(
    "Select prediction model",
    options=list(model_options.keys()),
)


left_column, right_column = st.columns(2)

with left_column:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=70,
        value=32,
        step=1,
    )

    gender = st.selectbox(
        "Gender",
        options=model_information["gender_options"],
    )

    education = st.selectbox(
        "Education Level",
        options=model_information["education_options"],
        index=(
            model_information["education_options"].index(
                "Bachelor's Degree"
            )
            if "Bachelor's Degree"
            in model_information["education_options"]
            else 0
        ),
    )

with right_column:
    years_experience = st.number_input(
        "Years of Experience",
        min_value=0.0,
        max_value=40.0,
        value=5.0,
        step=0.5,
    )

    default_job_title = (
        model_information["job_title_options"].index("Software Engineer")
        if "Software Engineer" in model_information["job_title_options"]
        else 0
    )

    job_title = st.selectbox(
        "Job Title",
        options=model_information["job_title_options"],
        index=default_job_title,
    )


if (
    age < model_information["age_min"]
    or age > model_information["age_max"]
    or years_experience < model_information["experience_min"]
    or years_experience > model_information["experience_max"]
):
    st.warning(
        "Age or experience is outside the clean training-data range. "
        "The prediction may be less reliable."
    )


if st.button("Predict Salary", type="primary"):
    selected_model_path = model_options[selected_model_name]
    model = load_saved_model(selected_model_path)

    input_data = pd.DataFrame(
        {
            "Age": [age],
            "Gender": [gender],
            "Education Level": [education],
            "Job Title": [job_title],
            "Years of Experience": [years_experience],
        }
    )

    predicted_salary = model.predict(input_data)[0]

    st.success("Prediction completed successfully.")
    st.metric(
        label="Estimated Salary",
        value=f"{predicted_salary:,.0f}",
    )
    st.caption(
        "The predicted value uses the same salary unit as the dataset."
    )


st.divider()
st.subheader("Model Comparison")

comparison = load_model_comparison().copy()
comparison = comparison.rename(columns={"R2_Score": "R² Score"})
comparison["R² Score"] = comparison["R² Score"].map(
    lambda value: f"{value:.4f}"
)
st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Random Forest gives the better test R² score on the clean dataset. "
    "Gradient Boosting is included to demonstrate boosting."
)


st.subheader("Download Trained Model")

selected_download_name = st.selectbox(
    "Choose model file",
    options=list(model_options.keys()),
    key="download_model",
)

download_path = model_options[selected_download_name]

with open(download_path, "rb") as model_file:
    st.download_button(
        label="Download Model Pipeline (.joblib)",
        data=model_file.read(),
        file_name=download_path.name,
        mime="application/octet-stream",
    )

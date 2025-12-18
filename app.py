"""
Streamlit app for predicting disease progression using a trained
Random Forest model on the sklearn diabetes dataset.
"""

import pickle

import pandas as pd
import streamlit as st
from sklearn.datasets import load_diabetes


MODEL_PATH = "model.pkl"


def load_reference_data():
    """
    Load the diabetes dataset as a DataFrame.
    We use this to get feature ranges and default values.
    """
    dataset = load_diabetes(as_frame=True)
    return dataset.frame


def load_trained_model(path=MODEL_PATH):
    """
    Load the trained model and feature names from the pickle file.
    Handles common errors in a user-friendly way.
    """
    try:
        with open(path, "rb") as f:
            model_bundle = pickle.load(f)
    except FileNotFoundError:
        st.error("Model file not found. Please run `python train_model.py` first.")
        st.stop()
    except Exception as exc:
        st.error(f"Unable to load model: {exc}")
        st.stop()

    # Expecting a dictionary with keys "model" and "feature_names"
    if "model" not in model_bundle or "feature_names" not in model_bundle:
        st.error("Model file is missing `model` or `feature_names` keys.")
        st.stop()

    model = model_bundle["model"]
    feature_names = model_bundle["feature_names"]
    return model, feature_names


def build_sidebar_inputs(feature_names, reference_df):
    """
    Create sidebar inputs for each feature and
    return a one-row DataFrame with the user values.
    """
    st.sidebar.header("Input Features")
    inputs = {}

    for name in feature_names:
        col = reference_df[name]
        min_val = float(col.min())
        max_val = float(col.max())
        default_val = float(col.mean())

        # The diabetes dataset is standardized: values are centered around 0.
        # For the `sex` feature (called Gender in the UI), use a dropdown mapped to the
        # encoded numeric values in the dataset.
        if name == "sex":
            unique_vals = sorted(col.unique())
            if len(unique_vals) == 2:
                label_map = {
                    "Female": float(unique_vals[0]),
                    "Male": float(unique_vals[1]),
                }
                default_label = "Female" if default_val <= 0 else "Male"
                selected_label = st.sidebar.selectbox(
                    "Gender",
                    options=list(label_map.keys()),
                    index=0 if default_label == "Female" else 1,
                    help=(
                        "Gender encoded as a standardized numeric value from the diabetes dataset."
                    ),
                )
                inputs[name] = label_map[selected_label]
                continue

        # Slider for numeric features
        step = max((max_val - min_val) / 200, 0.001)

        inputs[name] = st.sidebar.slider(
            label=name,
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            step=step,
            help=f"Value range in dataset: {min_val:.3f} to {max_val:.3f}",
        )

    # Return a single-row DataFrame with the features in the correct order
    return pd.DataFrame([inputs], columns=feature_names)


def main():
    st.set_page_config(page_title="Diabetes Progression Predictor", layout="wide")

    st.title("Diabetes Progression Predictor")
    st.write(
        "This app uses the classic **sklearn diabetes** dataset to predict "
        "disease progression one year after baseline based on ten standardized "
        "clinical features (age, gender, BMI, blood pressure, and six lab "
        "measurements). Adjust the inputs in the sidebar and click "
        "**Predict** to see the model's estimated progression score."
    )

    # Load data and model
    reference_df = load_reference_data()
    model, feature_names = load_trained_model()

    # Build the input row for prediction
    user_df = build_sidebar_inputs(feature_names, reference_df)

    st.sidebar.markdown("---")
    predict_clicked = st.sidebar.button("Predict", type="primary")

    if predict_clicked:
        try:
            prediction = model.predict(user_df)[0]
            st.success(
                f"Estimated disease progression score: {prediction:.2f}",
                icon="✅",
            )
            st.info(
                "Higher scores indicate a greater predicted progression over "
                "the next year.",
                icon="ℹ️",
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
    else:
        st.info("Set the inputs in the sidebar and click Predict to see a result.")


if __name__ == "__main__":
    main()



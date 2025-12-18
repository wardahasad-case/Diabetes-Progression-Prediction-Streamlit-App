"""
Train a Random Forest model on the sklearn diabetes dataset and
save the trained model to a file called `model.pkl`.
"""

import pickle

import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


MODEL_PATH = "model.pkl"
RANDOM_STATE = 42


def load_data():
    """
    Load the diabetes dataset and return:
    - X: input features (as a DataFrame)
    - y: target values (as a Series)
    - feature_names: list of feature column names
    """
    dataset = load_diabetes(as_frame=True)
    df = dataset.frame
    feature_names = list(dataset.feature_names)

    # Select the feature columns and the target column
    X = df[feature_names]
    y = df["target"]
    return X, y, feature_names


def train_model():
    """
    Train a Random Forest Regressor on the diabetes dataset.
    Prints simple evaluation metrics and returns the model and feature names.
    """
    X, y, feature_names = load_data()

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Create and train the model
    model = RandomForestRegressor(
        n_estimators=250,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=1,
    )
    model.fit(X_train, y_train)

    # Evaluate the model on the test data
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    # Some older sklearn versions do not have `squared` argument,
    # so we compute RMSE by hand from MSE.
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    print(f"Validation R2: {r2:.3f}")
    print(f"Validation RMSE: {rmse:.3f}")

    return model, feature_names


def save_model(model, feature_names, path=MODEL_PATH):
    """
    Save the trained model and its feature names to a pickle file.
    """
    with open(path, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "feature_names": feature_names,
            },
            f,
        )
    print(f"Model saved to {path}")


def main():
    model, feature_names = train_model()
    save_model(model, feature_names)


if __name__ == "__main__":
    main()



## Diabetes Progression Prediction – Streamlit App
<img width="1919" height="846" alt="image" src="https://github.com/user-attachments/assets/035b2992-1e52-4da9-a1d1-5ddecc63ceae" />


This project is a small end‑to‑end machine learning demo built with **Python**, **scikit‑learn**, and **Streamlit**.  
It trains a **Random Forest Regressor** on the classic `load_diabetes` dataset from `sklearn.datasets` and serves an interactive web UI to make predictions.

---

### Dataset: sklearn Diabetes

We use the **Diabetes dataset** included in scikit‑learn (`load_diabetes`).  
It contains data collected from diabetes patients and is commonly used for regression tasks.

- **Task type**: Regression (predict a continuous score)
- **Target**: A quantitative measure of **disease progression one year after baseline**
- **Number of samples**: 442
- **Number of input features**: 10 (all are standardized numeric features)

The features are:

- **age**: Age of the patient (standardized)
- **sex**: Biological sex of the patient (standardized, shown as **Gender** in the UI)
- **bmi**: Body Mass Index (standardized)
- **bp**: Average blood pressure (standardized)
- **s1–s6**: Six blood serum measurements (standardized lab values)

All features are **standardized** in this dataset, which is why typical values are around 0 (e.g., ‑0.05, 0.03, etc.), not raw units.

---

### Project Structure

- **`train_model.py`**  
  Loads the diabetes dataset, trains a `RandomForestRegressor`, evaluates it (R² and RMSE), and saves the trained model and feature names to **`model.pkl`**.

- **`app.py`**  
  Streamlit web application that:
  - Loads `model.pkl`
  - Shows a description of the dataset and prediction target
  - Uses a **sidebar** for user input (sliders and a Gender dropdown)
  - Predicts disease progression and displays the result in a highlighted box

- **`requirements.txt`**  
  Python dependencies for this project:
  - `streamlit`
  - `scikit-learn`
  - `pandas`

---

### How to Run the Project

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model and create `model.pkl`**:

   From the project root:

   ```bash
   python train_model.py
   ```

   You should see validation metrics (R² and RMSE) in the terminal and a message like:

   > Model saved to model.pkl

3. **Run the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

4. **Use the web UI**:

   - Open the URL shown in the terminal (usually `http://localhost:8501`).
   - Read the short description at the top.
   - Use the **sidebar** to adjust:
     - Age, BMI, blood pressure, and lab measurements (sliders)
     - Gender (dropdown)
   - Click **Predict** to see the estimated disease progression score.

---

### Model Details

The model used is a **Random Forest Regressor** from scikit‑learn:

- `n_estimators = 250`
- `random_state = 42`
- `test_size = 0.2` for the train/test split

The script prints:

- **R² score** on the test set (how well the model explains variance)
- **RMSE** (Root Mean Squared Error) as a measure of prediction error

You can re‑run `train_model.py` any time to retrain the model and overwrite `model.pkl`.

---





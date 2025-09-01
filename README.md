# 🚗 Car Price Prediction App

This web application uses machine learning to predict the selling price of a car. By analyzing various features like brand, engine specifications, body type, and fuel type, the trained model provides an estimated price, offering valuable insights for car buyers, sellers, and enthusiasts.

-----

## ⚙️ Key Features

  * **ML-based Price Prediction**: Enter a car from the dataset to get its predicted price based on our trained model.
  * **Actual vs. Predicted Comparison**: For any car in the dataset, you can see both its actual historical price and the model's prediction, illustrating the model's accuracy and expected deviations.
  * **Car Filtering/Search**: Instantly filter and search cars by brand, price range, or fuel type.
  * **Modern UI**: Built with **Streamlit** for a fast, interactive, and user-friendly experience.

-----

## 🛠️ Data Preparation & Processing

The project follows a standard machine learning workflow to prepare the data for training.

  * **Initial Cleaning**: We dropped non-essential columns like `car_ID` and `symboling`, and then extracted the car brand from the `CarName` column.
  * **Feature Engineering**: To enhance the model's predictive power, we created two new features:
      * `power_to_weight_ratio`: Calculated as `horsepower / curbweight`.
      * `fuel_efficiency_index`: The average of `citympg` and `highwaympg`.
  * **Categorical Encoding**: We used **one-hot encoding** to convert categorical features (e.g., `fueltype`, `carbody`, `CarBrand`) into a numerical format suitable for the model.
  * **Numerical Scaling**: All numerical features were transformed using **StandardScaler** to have a zero mean and unit variance, which improves model performance.
  * **Train/Test Split**: The dataset was split into an 80% training set and a 20% testing set to ensure an unbiased evaluation of the model's performance.

-----

## 🧠 Model Building

### Model Used

The core of this app is a **Random Forest Regressor**. This tree-based ensemble model is particularly effective at capturing the non-linear relationships and complex patterns present in car data.

### Hyperparameter Tuning

We used **grid search cross-validation** to find the optimal values for the key hyperparameters, `n_estimators` (the number of trees in the forest) and `max_depth` (the maximum depth of each tree).

### Evaluation Metrics

The model's performance is primarily measured using:

  * **R² Score**: The main metric, representing the proportion of variance in the price that the model can explain. A higher score is better.
  * **Mean Absolute Error (MAE)** and **Mean Squared Error (MSE)**: These metrics provide a sense of the average prediction error in dollar terms.

-----

## 📊 Model Accuracy & Expected Usage

The Random Forest model typically achieves a high R² score on the test data, often in the **0.85–0.93 range**, which means it can explain 85% to 93% of the price variance. This makes it a highly reliable tool for price estimation.

**Note**: The model does not predict the exact price for every car. It provides a very reasonable estimate that's usually close to the actual price but may be slightly above or below. This is normal for regression models and reflects the inherent complexity and unmodeled factors in real-world data. The app shows both the actual and predicted prices to demonstrate this expected behavior and build user confidence in the model's general utility.

-----

## 🚀 How to Run

1.  **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

2.  **Train the Model**:

    ```bash
    python train_model.py
    ```

    This script will preprocess the data, train the model, and save all necessary files (`best_car_price_model.pkl`, `scaler.pkl`, etc.).

3.  **Launch the App**:

    ```bash
    streamlit run app.py
    ```

    The app will open in your default web browser.





<img width="1919" height="906" alt="Screenshot 2025-09-01 210522" src="https://github.com/user-attachments/assets/e559bfd2-d2fa-4204-99ba-e90525a985f5" />

-----

## 💻 Project Structure

```
.
├── app.py                     # Streamlit web application
├── train_model.py             # Data processing and model training script
├── data.csv                   # The car dataset
├── best_car_price_model.pkl   # The trained machine learning model
├── scaler.pkl                 # The saved StandardScaler object
├── features.pkl               # List of features used by the model
├── numerical_cols.pkl         # List of numerical columns
├── requirements.txt           # Python dependencies
└── README.md
```

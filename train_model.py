import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the dataset
df = pd.read_csv("data.csv")

# --- Data Preparation & Feature Engineering ---
df = df.drop(['car_ID', 'symboling'], axis=1)
df['CarBrand'] = df['CarName'].apply(lambda x: x.split(' ')[0])
df = df.drop('CarName', axis=1)

categorical_cols = [
    'fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel',
    'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem', 'CarBrand'
]

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
df_encoded['power_to_weight_ratio'] = df_encoded['horsepower'] / df_encoded['curbweight']
df_encoded['fuel_efficiency_index'] = (df_encoded['citympg'] + df_encoded['highwaympg']) / 2

X = df_encoded.drop('price', axis=1)
y = df_encoded['price']

# Save numerical cols for inference
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# --- Model Building & Saving ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None]
}
rf_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1, scoring='r2')
grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_

# Save the model, scaler, feature list, and numerical columns
joblib.dump(best_rf_model, 'best_car_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(list(X.columns), 'features.pkl')
joblib.dump(numerical_cols, 'numerical_cols.pkl')

print("All files have been saved locally. They are now ready for your Streamlit app.")

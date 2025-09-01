import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model, scaler, features and numerical columns
try:
    model = joblib.load('best_car_price_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('features.pkl')
    numerical_cols = joblib.load('numerical_cols.pkl')
    df_raw = pd.read_csv('data.csv')
except FileNotFoundError:
    st.error("Model or data files missing. Ensure all files are present.")
    st.stop()

st.title("🚗 Car Finder & Price Predictor")
st.write("Filter cars by brand, fuel type, price or predict price for a specific car.")

# Sidebar filters
st.sidebar.header("Filter Cars")
df_raw['CarBrand'] = df_raw['CarName'].apply(lambda x: x.split(' ')[0])
car_brands = sorted(df_raw['CarBrand'].unique().tolist())
fuel_types = sorted(df_raw['fueltype'].unique().tolist())
min_price, max_price = st.sidebar.slider("Select Price Range",
                                        int(df_raw['price'].min()), int(df_raw['price'].max()),
                                        (int(df_raw['price'].min()), int(df_raw['price'].max())))

selected_brand = st.sidebar.selectbox("Car Brand", ['All'] + car_brands)
selected_fuel = st.sidebar.selectbox("Fuel Type", ['All'] + fuel_types)

# Filter data based on sidebar choices
filtered_df = df_raw.copy()
if selected_brand != 'All':
    filtered_df = filtered_df[filtered_df['CarBrand'] == selected_brand]
if selected_fuel != 'All':
    filtered_df = filtered_df[filtered_df['fueltype'] == selected_fuel]
filtered_df = filtered_df[(filtered_df['price'] >= min_price) & (filtered_df['price'] <= max_price)]

st.write(f"### Found {filtered_df.shape[0]} Matching Cars")
if not filtered_df.empty:
    st.dataframe(filtered_df[['CarName', 'fueltype', 'carbody', 'price']].reset_index(drop=True))
else:
    st.info("No cars match your criteria. Please adjust the filters.")
st.markdown("---")

# Price prediction section
st.header("Predict Price for a Car")
car_options = filtered_df['CarName'].unique()
if len(car_options) > 0:
    selected_car_name = st.selectbox("Select a specific car:", car_options)
    if selected_car_name:
        car_data = filtered_df[filtered_df['CarName'] == selected_car_name].iloc[0]

        # Build input dict based on features saved during training
        input_dict = {feature: 0 for feature in features}

        # Numerical features
        input_dict['horsepower'] = car_data['horsepower']
        input_dict['enginesize'] = car_data['enginesize']
        input_dict['curbweight'] = car_data['curbweight']
        input_dict['citympg'] = car_data['citympg']
        input_dict['highwaympg'] = car_data['highwaympg']
        input_dict['power_to_weight_ratio'] = car_data['horsepower'] / car_data['curbweight']
        input_dict['fuel_efficiency_index'] = (car_data['citympg'] + car_data['highwaympg']) / 2

        # One-hot encoded features
        car_brand = car_data['CarName'].split(' ')[0]
        onehot_features = {
            f'fueltype_{car_data["fueltype"]}',
            f'aspiration_{car_data["aspiration"]}',
            f'doornumber_{car_data["doornumber"]}',
            f'carbody_{car_data["carbody"]}',
            f'drivewheel_{car_data["drivewheel"]}',
            f'enginelocation_{car_data["enginelocation"]}',
            f'enginetype_{car_data["enginetype"]}',
            f'cylindernumber_{car_data["cylindernumber"]}',
            f'fuelsystem_{car_data["fuelsystem"]}',
            f'CarBrand_{car_brand}'
        }
        for col in onehot_features:
            if col in input_dict:
                input_dict[col] = 1

        # Create DataFrame and align columns
        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(features, axis=1, fill_value=0)

        # Scale numerical features only
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        prediction = model.predict(input_df)[0]
        actual_price = car_data['price']

        # Show both prices for clarity
        st.info(f"Actual price in dataset: **${actual_price:,.2f}**")
        st.success(f"Predicted price for the selected {selected_car_name}: **${prediction:,.2f}**")
else:
    st.info("No cars available for prediction. Adjust filters.")

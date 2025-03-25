# Importing the required libraries
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

print("California Housing Dataset:")
print(df.head())

# Feature and target variable
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error: {mse}")
print(f"R2 Score: {r2}")

print("\nModel Coefficients:")
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

coef_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coef_df)

# Test the model
new_data = pd.DataFrame({
    'MedInc': [3.0],
    'HouseAge': [30],
    'AveRooms': [6],
    'AveBedrms': [1],
    'Population': [500],
    'AveOccup': [3],
    'Latitude': [34.5],
    'Longitude': [-118.25]
})

predicted_price = model.predict(new_data)
print(f"\nPredicted House Price: ${predicted_price[0]:,.2f}")
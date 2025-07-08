# HOUSE-PRICE-PREDICTIONimport pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load dataset
data = pd.read_csv('data/housing.csv')  # Replace with your dataset

# Example features: location, bhk, bath, sqft
data = pd.get_dummies(data, columns=["location"], drop_first=True)
X = data.drop("price", axis=1)
y = data["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# Save model
with open("model/house_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save column info for prediction interface
columns = list(X.columns)
with open("model/columns.pkl", "wb") as f:
    pickle.dump(columns, f)

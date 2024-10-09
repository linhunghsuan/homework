# Step 1: Business Understanding
# Goal: Predict house prices based on house size

# Step 2: Data Understanding
# Generate a sample dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Data: House size (X) and house price (Y)
house_size = np.array([1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400])
house_price = np.array([300, 320, 340, 360, 380, 400, 420, 440, 460, 480])

# Convert to DataFrame for easier manipulation
df = pd.DataFrame({
    'Size': house_size,
    'Price': house_price
})

# Step 3: Data Preparation
# Split the data into training and test sets
X = df[['Size']]  # Features
y = df['Price']   # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Modeling
# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Evaluation
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation results
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Visualize the results
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Size (sq. ft.)')
plt.ylabel('Price ($1000s)')
plt.title('Linear Regression: House Size vs. Price')
plt.legend()
plt.show()

# Step 6: Deployment (Making predictions)
# Predict the price of a house of 2500 sq. ft.
new_size = np.array([[2500]])
predicted_price = model.predict(new_size)
print(f"The predicted price of a house with 2500 sq. ft. is ${predicted_price[0]:.2f}K")

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Set the app title
st.title('Interactive Linear Regression with Adjustable Parameters')

# Instructions
st.write("""
This app allows you to adjust the following parameters for a linear regression:
- Slope: The steepness of the line.
- Noise Scale: The randomness added to the data.
- Number of Points: The number of data points in the dataset.
""")

# Sidebar sliders to adjust parameters
slope = st.sidebar.slider('Slope', 0.0, 10.0, 1.0)
noise_scale = st.sidebar.slider('Noise Scale', 0.0, 100.0, 20.0)
n_points = st.sidebar.slider('Number of Points', 10, 500, 100)

# Generate random data based on user-defined slope, noise, and number of points
np.random.seed(42)
X = np.linspace(0, 100, n_points)
noise = np.random.normal(0, noise_scale, n_points)
y = slope * X + noise

# Reshape X for linear regression
X_reshaped = X.reshape(-1, 1)

# Train the linear regression model
model = LinearRegression()
model.fit(X_reshaped, y)

# Generate predictions
y_pred = model.predict(X_reshaped)

# Display the model's slope and intercept
st.write(f"### Linear Regression Model: ")
st.write(f"**Estimated Slope (Coefficient):** {model.coef_[0]:.4f}")
st.write(f"**Intercept:** {model.intercept_:.2f}")

# Plot the data points and the regression line
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Adjustable Parameters')
plt.legend()

# Display the plot
st.pyplot(plt)

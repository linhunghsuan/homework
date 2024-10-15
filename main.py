import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# ver 1.0.1 use crisp-dm
# ver 1.1.1 use Streamlit
# Set the app title
st.title('Interactive Linear Regression')

# Instructions
st.write("""
This app allows you to explore how changes in house size affect the predicted house price using linear regression.
""")

# Sidebar sliders to adjust the number of points and noise level
n_points = st.sidebar.slider('Number of Data Points', 5, 50, 10)
noise_level = st.sidebar.slider('Noise Level', 0.0, 100.0, 20.0)

# Generate a sample dataset based on the user input
np.random.seed(42)
house_size = np.linspace(1000, 3000, n_points)
house_price = 200 + house_size * 0.1 + np.random.normal(0, noise_level, n_points)

# Create a DataFrame to hold the data
df = pd.DataFrame({'Size': house_size, 'Price': house_price})

# Display the data
st.write('### Generated Data')
st.write(df)

# Prepare the data for modeling
X = df[['Size']]
y = df['Price']

# Train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Generate predictions
y_pred = model.predict(X)

# Display the slope and intercept of the regression line
st.write(f"### Linear Regression Model: ")
st.write(f"**Slope (Coefficient):** {model.coef_[0]:.4f}")
st.write(f"**Intercept:** {model.intercept_:.2f}")

# Plot the data points and the regression line
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('House Size (sq. ft.)')
plt.ylabel('House Price ($1000s)')
plt.title('House Size vs. Price')
plt.legend()

# Display the plot
st.pyplot(plt)

# Predict the price of a new house size using a slider
new_size = st.sidebar.slider('New House Size (sq. ft.)', 1000, 3000, 1500)
predicted_price = model.predict([[new_size]])

st.write(f"### Predicted Price for House Size {new_size} sq. ft.: ${predicted_price[0]:.2f}K")

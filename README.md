# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Load the dataset from a CSV file and separate the features and target variable, encoding any categorical variables as needed.<br>
2. Scale the features using a standard scaler to normalize the data.<br>
3. Initialize model parameters (theta) and add an intercept term to the feature set.<br>
4. Train the linear regression model using gradient descent by iterating through a specified number of iterations to minimize the cost function.<br>
5. Make predictions on new data by transforming it using the same scaling and encoding applied to the training data.

## Program:

```
Program to implement the linear regression using gradient descent.
Developed by: SWETHA K
RegisterNumber: 212224230284

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("exp_3_50_Startups.csv")

# Display first few rows
print("First 5 rows of the dataset:")
display(data.head())

# One-hot encode the 'State' column
data_encoded = pd.get_dummies(data, columns=['State'], drop_first=True)

# Display the encoded dataset
print("Encoded dataset:")
display(data_encoded.head())

# Independent features (all columns except Profit)
X_raw = data_encoded.drop('Profit', axis=1).values

# Target variable (Profit)
y_raw = data_encoded['Profit'].values.reshape(-1, 1)

print("Shape of features:", X_raw.shape)
print("Shape of target:", y_raw.shape)

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X = X_scaler.fit_transform(X_raw)
y = y_scaler.fit_transform(y_raw)

# Add a column of ones to X for bias term
m = X.shape[0]
X = np.hstack((np.ones((m, 1)), X))
print("Shape of X after adding bias term:", X.shape)

def compute_cost(X, y, theta):
    """
    Compute Mean Squared Error cost.
    """
    m = len(y)
    preds = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum((preds - y) ** 2)
    return cost

def gradient_descent(X, y, learning_rate=0.01, num_iters=2000, tol=1e-8, verbose=False):
    """
    Batch Gradient Descent for Linear Regression.
    """
    m, n = X.shape
    theta = np.zeros((n, 1))
    J_history = []

    prev_cost = compute_cost(X, y, theta)

    for i in range(num_iters):
        preds = X.dot(theta)
        errors = preds - y
        grad = (1 / m) * (X.T.dot(errors))
        theta -= learning_rate * grad

        cost = compute_cost(X, y, theta)
        J_history.append(cost)

        if abs(prev_cost - cost) < tol:
            if verbose:
                print(f"Converged at iteration {i}")
            break
        prev_cost = cost

        if verbose and (i % 500 == 0 or i < 5):
            print(f"Iteration {i:4d}, Cost: {cost:.6f}")

    return theta, J_history

alpha = 0.01
theta, J_hist = gradient_descent(X, y, learning_rate=alpha, num_iters=5000, tol=1e-9, verbose=True)

print("\nLearned Parameters (Theta):")
print(theta.flatten())

plt.figure(figsize=(7,4))
plt.plot(J_hist)
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE/2)')
plt.title('Cost Function Convergence')
plt.grid(True)
plt.show()

# Create new sample (same feature names as original)
new_sample = pd.DataFrame([{
    'R&D Spend': 165349.2,
    'Administration': 136897.8,
    'Marketing Spend': 471784.1,
    'State': 'New York'
}])

# Apply one-hot encoding
new_encoded = pd.get_dummies(new_sample, columns=['State'], drop_first=True)

# Align columns with training data (add missing ones as 0)
new_encoded = new_encoded.reindex(columns=data_encoded.drop('Profit', axis=1).columns, fill_value=0)

# Scale new data
new_scaled = X_scaler.transform(new_encoded)

# Add bias
new_design = np.hstack((np.ones((new_scaled.shape[0], 1)), new_scaled))

# Predict (scaled)
scaled_pred = new_design.dot(theta)

# Inverse transform to original units
pred_original = y_scaler.inverse_transform(scaled_pred)

print(f"\nPredicted Profit for the new startup: ₹{pred_original[0][0]:,.2f}")

```

## Output:


<img width="801" height="285" alt="image" src="https://github.com/user-attachments/assets/8af82891-00e8-480f-a678-9456d4e77a05" />
<img width="971" height="280" alt="image" src="https://github.com/user-attachments/assets/52e48596-5f59-4f4c-b1cb-72129993672a" />
<img width="542" height="67" alt="image" src="https://github.com/user-attachments/assets/c2a84880-2ad9-42fe-9e08-c59918d4225b" />
<img width="806" height="37" alt="image" src="https://github.com/user-attachments/assets/40805508-e6d0-4d95-b357-0de6122a68c4" />
<img width="1052" height="355" alt="image" src="https://github.com/user-attachments/assets/47f5dd08-3ea1-4247-be00-3593c87cbd9a" />
<img width="1077" height="523" alt="image" src="https://github.com/user-attachments/assets/8d8fd4c3-cb44-4947-a7fa-523da3669aa7" />
<img width="1698" height="93" alt="image" src="https://github.com/user-attachments/assets/9d68f9c8-66a0-47e7-90b1-0d7de2e0074d" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.

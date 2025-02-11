1. Data Preparation
The dataset is read from a CSV file containing package data, including the number of packages (total_pkgs) and date (swak_dt).
Date Parsing: The swak_dt column is converted to datetime format.
Aggregation: The data is aggregated by week, summing the total number of packages per week.
Filtering: The dataset is filtered to include data between May 1, 2018, and September 23, 2024.
Missing Data Handling: Any missing weekly data points are filled with zeros.

2. Feature Engineering
Several new features are created to improve the model's predictive power:
Week of Year: The week number of the year is added.
Holiday Flag: A binary flag is added to mark US Federal Holidays.
Lag Features: The model uses lag features (up to 14 weeks) to capture previous data points’ influence.
Rolling Statistics: Moving averages and rolling standard deviations are computed over 7 and 14-week windows.

3. Feature Scaling
Feature scaling is performed using StandardScaler from sklearn to standardize the features, making them comparable and improving model performance.

4. Sequence Creation for LSTM
The dataset is transformed into sequences of historical data, where each sequence represents a specific number of past weeks (52 weeks) to predict the target (the next 52 weeks).

5. Train-Test Split
The data is split into training and testing sets (80% for training and 20% for testing). The training data is further converted into PyTorch tensors to be fed into the LSTM model.

6. Model Definition (LSTM + RNN)
An LSTM model is defined using PyTorch. The model consists of:
Bidirectional LSTM: This allows the model to learn from both the past and future (relative to each point in the sequence).
Fully Connected Layer: After LSTM, the output is passed through a fully connected layer to predict the target value.

7. Model Training with Early Stopping and Gradient Clipping
Training Loop: The model is trained over multiple epochs, with the loss being calculated using Mean Squared Error (MSE) and the optimizer being Adam.
Gradient Clipping: To avoid exploding gradients, the gradients are clipped to a maximum norm.
Early Stopping: If validation loss does not improve over a certain number of epochs, training stops early to prevent overfitting.

8. Model Evaluation
After training, the model is evaluated using the test set. Various performance metrics are computed, including:
MSE (Mean Squared Error)
MAE (Mean Absolute Error)
R² (R-squared): Measures how well the model fits the data.
MAPE (Mean Absolute Percentage Error): Measures accuracy as a percentage.
sMAPE (Symmetric Mean Absolute Percentage Error): A variation of MAPE that accounts for zero values in the data.

9. Plotting Predictions
The true and predicted values are plotted on a graph for visual comparison.
Requirements
numpy
pandas
torch
matplotlib
sklearn

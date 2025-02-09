import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset from the specific sheet
file_path = '2023_Energy data ERP_COMPLETE.xlsx'  # Update with the correct path
sheet_name = 'Weekly (TOTAL) (1)'  # Update the sheet name

# Load the data
data = pd.read_excel(file_path, sheet_name=sheet_name, header=0)

# Rename the first two columns to 'Buildings' and 'Module'
data.rename(columns={'Unnamed: 0': 'Buildings', 'Unnamed: 1': 'Module'}, inplace=True)

# Display the first few rows to understand the structure
print("Initial data:")
print(data.head())

# Step 1: Melt the dataset to transform weekly data into a long format
melted_data = pd.melt(
    data,
    id_vars=['Buildings', 'Module'],  # Identifier columns
    var_name='Week',                  # Weekly consumption columns
    value_name='Energy Consumption'   # Energy consumption values
)

# Clean the 'Week' column to extract only the week range as a feature
print("Melted data (first few rows):")
print(melted_data.head())

# Add date-based features (e.g., week number)
melted_data['Week Number'] = melted_data['Week'].str.extract(r'(\d+)').astype(int)  # Extract week numbers
melted_data['Energy Consumption'] = pd.to_numeric(melted_data['Energy Consumption'], errors='coerce')

# Drop rows with missing values
melted_data = melted_data.dropna(subset=['Energy Consumption'])

# Step 2: Select Features and Target
# Use 'Week Number' as the primary feature
features = ['Week Number']
X = melted_data[features].values
y = melted_data['Energy Consumption'].values

# Normalize the data for machine learning models
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

# Reshape features for LSTM (samples, timesteps, features)
X_lstm = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train_lstm, X_test_lstm, y_train_scaled, y_test_scaled = train_test_split(X_lstm, y_scaled, test_size=0.2, random_state=42)

# Step 3: Train Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Step 4: Train Random Forest
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)
y_pred_rf = random_forest_model.predict(X_test)

# Step 5: Train Gradient Boosting
gradient_boosting_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gradient_boosting_model.fit(X_train, y_train)
y_pred_gb = gradient_boosting_model.predict(X_test)

# Step 6: Train LSTM
lstm_model = Sequential()
lstm_model.add(LSTM(units=100, return_sequences=True, input_shape=(1, X_train_lstm.shape[2])))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=50, return_sequences=False))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(25, activation='relu'))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lstm_model.fit(X_train_lstm, y_train_scaled, epochs=100, batch_size=16, validation_data=(X_test_lstm, y_test_scaled), callbacks=[early_stop], verbose=1)

y_pred_lstm_scaled = lstm_model.predict(X_test_lstm)
y_pred_lstm = scaler.inverse_transform(y_pred_lstm_scaled)

# Step 7: Evaluation Metrics
models = {
    'Linear Regression': y_pred_linear,
    'Random Forest': y_pred_rf,
    'Gradient Boosting': y_pred_gb,
    'LSTM': y_pred_lstm.flatten()
}

for name, y_pred in models.items():
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} - MSE: {mse:.4f}, R²: {r2:.4f}")

# Step 8: Visualization
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='True Energy Consumption', color='blue', linestyle='--')
plt.plot(y_pred_linear, label='Linear Regression', color='green')
plt.plot(y_pred_rf, label='Random Forest', color='red')
plt.plot(y_pred_gb, label='Gradient Boosting', color='purple')
plt.plot(y_pred_lstm, label='LSTM', color='orange')
plt.title('Model Predictions vs True Energy Consumption')
plt.xlabel('Test Samples')
plt.ylabel('Energy Consumption (kWh/m²)')
plt.legend()
plt.tight_layout()
plt.show()
# Step 8: Enhanced Visualization with Different Line Styles
plt.figure(figsize=(12, 8))

# Plot True Energy Consumption
plt.plot(range(len(y_test)), y_test, label='True Energy Consumption', color='blue', linestyle='--', marker='o')

# Plot predictions with different line styles and markers
plt.plot(range(len(y_pred_linear)), y_pred_linear, label='Linear Regression', color='green', linestyle='-', marker='x')
plt.plot(range(len(y_pred_rf)), y_pred_rf, label='Random Forest', color='red', linestyle='-.', marker='s')
plt.plot(range(len(y_pred_gb)), y_pred_gb, label='Gradient Boosting', color='purple', linestyle=':', marker='v')
plt.plot(range(len(y_pred_lstm)), y_pred_lstm, label='LSTM', color='orange', linestyle='-', marker='d')

# Add title and labels
plt.title('Enhanced Model Predictions vs True Energy Consumption')
plt.xlabel('Test Samples')
plt.ylabel('Energy Consumption (kWh/m²)')
plt.legend()

# Add grid for better readability
plt.grid(True, linestyle='--', linewidth=0.5)

# Show the plot
plt.tight_layout()
plt.show()


#--------------------------
from sklearn.neighbors import KNeighborsRegressor

# Step 3: Replace Linear Regression with K-Nearest Neighbors
knn_model = KNeighborsRegressor(n_neighbors=5)  # Simple KNN model with 5 neighbors
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

# Evaluation Metrics for KNN
mse_knn = mean_squared_error(y_test, y_pred_knn)
r2_knn = r2_score(y_test, y_pred_knn)
print(f"KNN - MSE: {mse_knn:.4f}, R²: {r2_knn:.4f}")

# Update the models dictionary to include KNN
models = {
    'KNN': y_pred_knn,
    'Random Forest': y_pred_rf,
    'Gradient Boosting': y_pred_gb,
    'LSTM': y_pred_lstm.flatten()
}

# Step 4: Enhanced Visualization with Week Numbers on X-Axis
plt.figure(figsize=(12, 8))

# Plot predictions with different line styles and markers
for name, y_pred in models.items():
    plt.plot(
        range(len(y_pred)),
        y_pred,
        label=name,
        linestyle='-',
        marker='o'
    )

# Add True Energy Consumption
plt.plot(range(len(y_test)), y_test, label='True Energy Consumption', color='blue', linestyle='--', marker='o', linewidth=2)

# Create x-ticks using the week numbers in the test set
week_numbers_test = melted_data.iloc[X_test.flatten().astype(int), :]['Week Number']
plt.xticks(
    ticks=range(len(week_numbers_test)),
    labels=week_numbers_test.values,
    rotation=45
)

# Add labels, title, legend, and grid
plt.title('Model Predictions vs True Energy Consumption (Weeks in Test Set)')
plt.xlabel('Week Number')
plt.ylabel('Energy Consumption (kWh/m²)')
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)

# Show the plot
plt.tight_layout()
plt.show()

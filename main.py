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
#
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Define file paths
# file_2022 = "2022_Energy data ERP_COMPLETE.xlsx"
# file_2023 = "2023_Energy data ERP_COMPLETE.xlsx"
#
# sheet_heating = 'Daily (HEATING) (2)'
# sheet_cooling = 'Daily (COOLING) (4)'
#
# # Check if files exist
# print(f"2022 File Exists: {os.path.exists(file_2022)}")
# print(f"2023 File Exists: {os.path.exists(file_2023)}")
#
# # Load data for both years
# try:
#     data_2022_heating = pd.read_excel(file_2022, sheet_name=sheet_heating)
#     data_2022_cooling = pd.read_excel(file_2022, sheet_name=sheet_cooling)
#     data_2023_heating = pd.read_excel(file_2023, sheet_name=sheet_heating)
#     data_2023_cooling = pd.read_excel(file_2023, sheet_name=sheet_cooling)
#     print("Data successfully loaded!")
# except Exception as e:
#     print(f"Error loading data: {e}")
#     exit()
#
# # Display initial data for verification
# print("2022 Heating Data:")
# print(data_2022_heating.head())
# print("\n2023 Heating Data:")
# print(data_2023_heating.head())
#
# print("\n2022 Cooling Data:")
# print(data_2022_cooling.head())
# print("\n2023 Cooling Data:")
# print(data_2023_cooling.head())
#
# # Drop unnecessary columns in Cooling Data if they exist
# def clean_cooling_data(df):
#     df.columns = df.columns.map(str)  # Convert all column names to strings
#     columns_to_drop = [col for col in df.columns if 'Unnamed' in col]
#     return df.drop(columns=columns_to_drop, errors='ignore')
#
# data_2022_cooling = clean_cooling_data(data_2022_cooling)
# data_2023_cooling = clean_cooling_data(data_2023_cooling)
#
# # Convert columns to datetime format where applicable
# def convert_columns_to_datetime(df):
#     for col in df.columns[2:]:  # Assuming first two columns are non-date columns
#         try:
#             df.rename(columns={col: pd.to_datetime(col, errors='coerce')}, inplace=True)
#         except Exception as e:
#             print(f"Could not convert column {col}: {e}")
#     return df
#
# data_2022_heating = convert_columns_to_datetime(data_2022_heating)
# data_2022_cooling = convert_columns_to_datetime(data_2022_cooling)
# data_2023_heating = convert_columns_to_datetime(data_2023_heating)
# data_2023_cooling = convert_columns_to_datetime(data_2023_cooling)
#
# # Example Plot: Heating data for "Module - A" in 2022 and 2023
# module_a_heating_2022 = data_2022_heating.iloc[0, 2:]  # Exclude non-date columns
# module_a_heating_2022.index = pd.to_datetime(module_a_heating_2022.index, errors='coerce')  # Ensure index is datetime
#
# module_a_heating_2023 = data_2023_heating.iloc[0, 2:]  # Exclude non-date columns
# module_a_heating_2023.index = pd.to_datetime(module_a_heating_2023.index, errors='coerce')  # Ensure index is datetime
#
# plt.figure(figsize=(12, 6))
# module_a_heating_2022.plot(label="2022", title="Heating Data for Module - A (2022 vs 2023)", xlabel="Date", ylabel="Heating Consumption")
# module_a_heating_2023.plot(label="2023")
# plt.legend()
# plt.grid()
# plt.show()
#
#
#
#
# # Save cleaned data
# data_2022_heating.to_csv("cleaned_heating_data_2022.csv", index=False)
# data_2022_cooling.to_csv("cleaned_cooling_data_2022.csv", index=False)
# data_2023_heating.to_csv("cleaned_heating_data_2023.csv", index=False)
# data_2023_cooling.to_csv("cleaned_cooling_data_2023.csv", index=False)
#
# print("Cleaned data saved as CSV files.")
#
#
# import os
# import pandas as pd
# import matplotlib.pyplot as plt

# Define file paths
file_2022 = "2022_Energy data ERP_COMPLETE.xlsx"
file_2023 = "2023_Energy data ERP_COMPLETE.xlsx"

sheet_heating = 'Daily (HEATING) (2)'
sheet_cooling = 'Daily (COOLING) (4)'

# Check if files exist
print(f"2022 File Exists: {os.path.exists(file_2022)}")
print(f"2023 File Exists: {os.path.exists(file_2023)}")

# Load data for both years
try:
    data_2022_heating = pd.read_excel(file_2022, sheet_name=sheet_heating)
    data_2022_cooling = pd.read_excel(file_2022, sheet_name=sheet_cooling)
    data_2023_heating = pd.read_excel(file_2023, sheet_name=sheet_heating)
    data_2023_cooling = pd.read_excel(file_2023, sheet_name=sheet_cooling)
    print("Data successfully loaded!")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Drop unnecessary columns in Cooling Data if they exist
def clean_cooling_data(df):
    df.columns = df.columns.map(str)  # Convert all column names to strings
    columns_to_drop = [col for col in df.columns if 'Unnamed' in col]
    return df.drop(columns=columns_to_drop, errors='ignore')

data_2022_cooling = clean_cooling_data(data_2022_cooling)
data_2023_cooling = clean_cooling_data(data_2023_cooling)

# Ensure date columns are aligned and preserved
def ensure_correct_date_columns(df):
    df = df.copy()
    # Ensure all columns from the 3rd onward are treated as dates
    for col in df.columns[2:]:  # Assuming first two columns are metadata
        try:
            df.rename(columns={col: pd.to_datetime(col, errors='coerce')}, inplace=True)
        except Exception as e:
            print(f"Error converting column {col}: {e}")
    return df

data_2022_heating = ensure_correct_date_columns(data_2022_heating)
data_2023_heating = ensure_correct_date_columns(data_2023_heating)
data_2022_cooling = ensure_correct_date_columns(data_2022_cooling)
data_2023_cooling = ensure_correct_date_columns(data_2023_cooling)

# Ensure only valid dates for 2022 and 2023 are retained
def filter_valid_years(df, start_year, end_year):
    df = df.copy()
    valid_columns = [col for col in df.columns if isinstance(col, pd.Timestamp) and start_year <= col.year <= end_year]
    return pd.concat([df.iloc[:, :2], df[valid_columns]], axis=1)

data_2022_heating = filter_valid_years(data_2022_heating, 2022, 2022)
data_2023_heating = filter_valid_years(data_2023_heating, 2023, 2023)
data_2022_cooling = filter_valid_years(data_2022_cooling, 2022, 2022)
data_2023_cooling = filter_valid_years(data_2023_cooling, 2023, 2023)

# Example Plot: Heating data for "Module - A" in 2022 and 2023
module_a_heating_2022 = data_2022_heating.iloc[0, 2:]  # Exclude metadata
module_a_heating_2022.index = pd.to_datetime(module_a_heating_2022.index, errors='coerce')

module_a_heating_2023 = data_2023_heating.iloc[0, 2:]  # Exclude metadata
module_a_heating_2023.index = pd.to_datetime(module_a_heating_2023.index, errors='coerce')

plt.figure(figsize=(12, 6))
module_a_heating_2022.plot(label="2022 Heating", title="Heating Data for Module - A (2022 vs 2023)", xlabel="Date", ylabel="Heating Consumption")
module_a_heating_2023.plot(label="2023 Heating")
plt.legend()
plt.grid()
plt.show()

# Example Plot: Cooling data for "Module - A" in 2022 and 2023
module_a_cooling_2022 = data_2022_cooling.iloc[0, 2:]  # Exclude metadata
module_a_cooling_2022.index = pd.to_datetime(module_a_cooling_2022.index, errors='coerce')

module_a_cooling_2023 = data_2023_cooling.iloc[0, 2:]  # Exclude metadata
module_a_cooling_2023.index = pd.to_datetime(module_a_cooling_2023.index, errors='coerce')

plt.figure(figsize=(12, 6))
module_a_cooling_2022.plot(label="2022 Cooling", title="Cooling Data for Module - A (2022 vs 2023)", xlabel="Date", ylabel="Cooling Consumption")
module_a_cooling_2023.plot(label="2023 Cooling")
plt.legend()
plt.grid()
plt.show()

# Merge data and save cleaned files
def merge_and_save_data(data1, data2, output_filename):
    merged_data = pd.concat([data1, data2], axis=0, ignore_index=True)
    merged_data.to_csv(output_filename, index=False)
    return merged_data

combined_heating = merge_and_save_data(data_2022_heating, data_2023_heating, "Combined_Heating_2022_2023.csv")
combined_cooling = merge_and_save_data(data_2022_cooling, data_2023_cooling, "Combined_Cooling_2022_2023.csv")

print("Combined data saved successfully.")

# Verification
print("Combined Heating Data (first 5 rows):")
print(combined_heating.head())
print("\nCombined Cooling Data (first 5 rows):")
print(combined_cooling.head())

#IE2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os


class EnergyAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.processed_data = None
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()

    def load_data(self):
        """Load data from Excel file"""
        try:
            print(f"Loading data from {self.file_path}")
            self.data = pd.read_excel(self.file_path)
            print("Data loaded successfully")
            print(f"Shape: {self.data.shape}")
            print(f"Columns: {self.data.columns.tolist()}")
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False

    def process_data(self):
        """Process the data into daily format"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return None

        # Melt the dataframe to get daily format
        # Assuming first two columns are 'Urban Area' and 'Building'
        id_vars = ['Urban Area', 'Building']
        value_vars = [col for col in self.data.columns if col not in id_vars]

        self.processed_data = pd.melt(
            self.data,
            id_vars=id_vars,
            value_vars=value_vars,
            var_name='Date',
            value_name='Consumption'
        )

        # Convert consumption to numeric
        self.processed_data['Consumption'] = pd.to_numeric(
            self.processed_data['Consumption'],
            errors='coerce'
        )

        # Extract day and month from date
        self.processed_data['Day'] = self.processed_data['Date'].str.extract('(\d+)').astype(int)
        month_map = {
            'gen': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'mag': 5, 'giu': 6,
            'lug': 7, 'ago': 8, 'set': 9, 'ott': 10, 'nov': 11, 'dic': 12
        }
        self.processed_data['Month'] = self.processed_data['Date'].str.extract('-(\w+)')[0].map(month_map)

        # Create categorical codes
        self.processed_data['Building_Code'] = pd.Categorical(self.processed_data['Building']).codes
        self.processed_data['Urban_Area_Code'] = pd.Categorical(self.processed_data['Urban Area']).codes

        return self.processed_data

    def analyze_patterns(self):
        """Analyze consumption patterns"""
        if self.processed_data is None:
            print("No processed data available. Please process data first.")
            return None

        # Daily patterns
        daily_stats = self.processed_data.groupby('Day')['Consumption'].agg([
            'mean', 'std', 'min', 'max'
        ]).round(3)

        # Building patterns
        building_stats = self.processed_data.groupby('Building')['Consumption'].agg([
            'mean', 'std', 'min', 'max'
        ]).round(3)

        # Urban area patterns
        area_stats = self.processed_data.groupby('Urban Area')['Consumption'].agg([
            'mean', 'std', 'min', 'max'
        ]).round(3)

        # Monthly patterns
        monthly_stats = self.processed_data.groupby('Month')['Consumption'].agg([
            'mean', 'std', 'min', 'max'
        ]).round(3)

        return {
            'daily_stats': daily_stats,
            'building_stats': building_stats,
            'area_stats': area_stats,
            'monthly_stats': monthly_stats
        }

    def create_visualizations(self):
        """Create visualizations of the consumption patterns"""
        if self.processed_data is None:
            print("No processed data available. Please process data first.")
            return

        plt.style.use('seaborn')

        # Figure 1: Daily Consumption Pattern
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        daily_avg = self.processed_data.groupby('Day')['Consumption'].mean()
        plt.plot(daily_avg.index, daily_avg.values)
        plt.title('Average Daily Energy Consumption')
        plt.xlabel('Day of Month')
        plt.ylabel('Consumption')

        # Figure 2: Monthly Pattern
        plt.subplot(2, 2, 2)
        sns.boxplot(data=self.processed_data, x='Month', y='Consumption')
        plt.title('Monthly Energy Consumption Distribution')
        plt.xlabel('Month')
        plt.ylabel('Consumption')

        # Figure 3: Urban Area Comparison
        plt.subplot(2, 2, 3)
        sns.boxplot(data=self.processed_data, x='Urban Area', y='Consumption')
        plt.xticks(rotation=45)
        plt.title('Energy Consumption by Urban Area')
        plt.xlabel('Urban Area')
        plt.ylabel('Consumption')

        # Figure 4: Building Comparison
        plt.subplot(2, 2, 4)
        building_avg = self.processed_data.groupby('Building')['Consumption'].mean().sort_values(ascending=False)
        plt.bar(range(len(building_avg)), building_avg.values)
        plt.title('Average Consumption by Building')
        plt.xticks(range(len(building_avg)), building_avg.index, rotation=90)
        plt.ylabel('Average Consumption')

        plt.tight_layout()
        plt.show()

    def train_model(self):
        """Train prediction model"""
        if self.processed_data is None:
            print("No processed data available. Please process data first.")
            return None

        # Prepare features
        features = ['Building_Code', 'Urban_Area_Code', 'Day', 'Month']
        X = self.processed_data[features]
        y = self.processed_data['Consumption']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model.fit(X_train_scaled, y_train)
        y_pred = self.model.predict(X_test_scaled)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Feature importance
        importance = pd.DataFrame({
            'feature': features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return {
            'mse': mse,
            'r2': r2,
            'importance': importance,
            'y_test': y_test,
            'y_pred': y_pred
        }


def main():
    # Initialize analyzer with the Excel file
    file_path = "2023_Energy data ERP_COMPLETE.xlsx"
    analyzer = EnergyAnalyzer(file_path)

    # Load data
    if not analyzer.load_data():
        return

    # Process data
    processed_data = analyzer.process_data()
    if processed_data is None:
        return

    # Analyze patterns
    patterns = analyzer.analyze_patterns()

    # Print statistics
    print("\nDaily Statistics:")
    print(patterns['daily_stats'])

    print("\nBuilding Statistics:")
    print(patterns['building_stats'])

    print("\nUrban Area Statistics:")
    print(patterns['area_stats'])

    print("\nMonthly Statistics:")
    print(patterns['monthly_stats'])

    # Create visualizations
    analyzer.create_visualizations()

    # Train model and get predictions
    model_results = analyzer.train_model()

    if model_results:
        print("\nModel Performance:")
        print(f"Mean Squared Error: {model_results['mse']:.4f}")
        print(f"R² Score: {model_results['r2']:.4f}")

        print("\nFeature Importance:")
        print(model_results['importance'])

    return analyzer, processed_data, patterns, model_results


if __name__ == "__main__":
    analyzer, processed_data, patterns, model_results = main()

# #Execution recommended in jupyter notebook

#!pip install timesfm[pax]

# Import timesfm library
import timesfm

# Initialize the TimesFM model with specified parameters
tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend="cpu", #Must be adjusted if necessary
        per_core_batch_size=32,
        context_len=416,
        horizon_len=128,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
      ),
    # Load the pretrained model checkpoint
      checkpoint=timesfm.TimesFmCheckpoint(
          huggingface_repo_id="google/timesfm-1.0-200m"),
)

import pandas as pd
import numpy as np
from collections import defaultdict
data = pd.read_csv('Electricity.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data[-544:] #because i only use 416+128 data points

#data = data.set_index('Date')
from sklearn.preprocessing import MinMaxScaler
# Normalize data, not needed for MAPE, and change all scaled_df to data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
scaled_df = pd.DataFrame(scaled_data, columns=['Value', 'temp'])
scaled_df['Date'] = pd.date_range(start='2018-07-06', periods=len(scaled_df), freq='D')
columns_order = ['Date'] + [col for col in scaled_df.columns if col != 'Date']
scaled_df = scaled_df[columns_order]

df = pd.DataFrame({'unique_id':[1]*len(data),'ds': data["Date"], "y":data['Value'], "temp":data['temp']}) #data = scaled_df

# Spliting data frame into train and test set
split_idx = int(len(df) * 0.766)
# Split the dataframe into train = historical data and test sets
train_df = df[:split_idx]
test_df = df[split_idx:]
print(train_df.shape, test_df.shape)

def get_full_data(context_len: int, horizon_len: int):
    # Ensure enough data is present for context and horizon
    if len(train_df) < context_len:
        raise ValueError(f"Dataset must have at least {context_len} rows for context.")

    if len(train_df) < context_len + horizon_len:
        print("Not enough data for the specified horizon length. Padding with zeros.")

    # Extract or pad context
    inputs = train_df["y"][:context_len].tolist()
    if len(inputs) < context_len:
        inputs += [0] * (context_len - len(inputs))  # Pad with zeros if necessary
    print(inputs)

    # Extract or pad covariates
    temp = data["temp"][:context_len + horizon_len].tolist()
    if len(temp) < context_len + horizon_len:
        temp += [0] * ((context_len + horizon_len) - len(temp))  # Pad with zeros
    print(temp)

    # Return data in the correct structure
    return {
        "inputs": [inputs],  # Single example with context length
        "temp": [temp],      # Covariate length includes horizon
    }

# Define context and horizon lengths
context_len = 416
horizon_len = 128

# Get prepared data
full_data = get_full_data(context_len=context_len, horizon_len=horizon_len)

# Debugging: Verify lengths
print(f"Inputs length: {len(full_data['inputs'][0])}")  # Should be 416
print(f"Temp length: {len(full_data['temp'][0])}")      # Should be (416 + 128)

# Run forecast with covariates
try:
    cov_forecast, ols_forecast = tfm.forecast_with_covariates(
        inputs=full_data["inputs"],  # Context data
        dynamic_numerical_covariates={
            "temp": full_data["temp"],  # Covariates (includes horizon)
        },
        freq=[0] * len(full_data["inputs"]),  # Frequency mask
        xreg_mode="xreg + timesfm",           # Mode for external regressors
        ridge=0.0,                            # Regularization
        force_on_cpu=False,                   # Use GPU if available
        normalize_xreg_target_per_input=True, # Normalize covariates
    )
    print("Forecast completed successfully!")
    print("Covariate Forecast:", cov_forecast)
    print("OLS Forecast:", ols_forecast)
except ValueError as e:
    print(f"Error during forecasting: {e}")

#set up graph
# Flatten all arrays inside cov_forecast into a single list
flattened_values = [item for sublist in cov_forecast for item in sublist]
# Convert the flattened list into a pandas Series or DataFrame
flattened_df = pd.DataFrame(flattened_values, columns=["Value"])
flattened_df['Date'] = pd.date_range(start='2019-08-26', periods=len(flattened_df), freq='D')

# Let's Visualise the Data
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')  # Setting the warnings to be ignored
# Set the style for seaborn
sns.set(style="darkgrid")
# Plot size
plt.figure(figsize=(15, 6))
# Plot forecasted values
sns.lineplot(x="Date", y='Value', data=flattened_df, color='red', label='Forecast')
# Plot Actual Time Series
sns.lineplot(x="ds", y='y', data=df, color='green', label='Actual Time Series')
plt.xlabel('Date')
plt.ylabel('Electric Production')
# Show the legend
plt.legend()
# Display the plot
plt.savefig('For_Cov_Electricity_plot.png')
plt.show()

actuals = test_df['y'] # add .iloc[-60:] for the last 60 datapoints, and change for the 7 and 30 day forecast to [:7]/[:30], no need to run the entire code again, only the part which calculates the errors
predicted_values = flattened_df['Value'] # same as above and always parallel
# Convert to numpy arrays
actual_values = np.array(actuals)
predicted_values = np.array(predicted_values)
# Calculate error metrics
MAE = np.mean(np.abs(actual_values - predicted_values))  # Mean Absolute Error
MSE = np.mean((actual_values - predicted_values)**2)     # Mean Squared Error
RMSE = np.sqrt(np.mean((actual_values - predicted_values)**2))  # Root Mean Squared Error
MAPE = np.mean(np.abs((actual_values - predicted_values) / actual_values))  # MAPE

# Print the error metrics
print(f"Mean Absolute Error (MAE): {MAE}")
print(f"Mean Squared Error (MSE): {MSE}")
print(f"Root Mean Squared Error (RMSE): {RMSE}")
print(f"Mean Absolute Percentage Error (MAPE): {MAPE}")

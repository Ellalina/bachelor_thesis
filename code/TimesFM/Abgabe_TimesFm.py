#!pip install timesfm[pax]

# Execution recommended in jupyter notebook

import timesfm

import pandas as pd
data=pd.read_csv('/content/total_consumption.csv')
data

data['Date']=pd.to_datetime(data['Date']) #convert to year-month-day
data.head()

data = data.set_index('Date')
from sklearn.preprocessing import MinMaxScaler
# Normalize data, not needed for MAPE, and change all scaled_df to data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
scaled_df = pd.DataFrame(scaled_data, columns=['Value', 'temp'])
scaled_df['Date'] = pd.date_range(start='2015-01-01', periods=len(scaled_df), freq='D')
columns_order = ['Date'] + [col for col in scaled_df.columns if col != 'Date']
scaled_df = scaled_df[columns_order]

df = pd.DataFrame({'unique_id':[1]*len(scaled_df),'ds': scaled_df["Date"], "y":scaled_df['Value'], 'temp':scaled_df['temp']})
df = df[-544:] #because i only use 416+128 data points

# Spliting data frame into train and test set
split_idx = int(len(df) * 0.766)
# Split the dataframe into train = historical data and test sets
train_df = df[:split_idx]
test_df = df[split_idx:]
print(train_df.shape, test_df.shape)


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
# Forecasting the values using the TimesFM model
timesfm_forecast = tfm.forecast_on_df(
   inputs=train_df,       # Input training data for training
   freq="D",              # Frequency of the time-series data (daily)
   value_name="y",        # Name of the column containing the values to be forecasted
   num_jobs=-1,           # Set to -1 to use all available cores
)
timesfm_forecast = timesfm_forecast[["ds","timesfm"]]


# Let's Visualise the Data
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')  # Setting the warnings to be ignored

scaled_df = scaled_df[-512:]
# Set the style for seaborn
sns.set(style="darkgrid")
# Plot size
plt.figure(figsize=(15, 6))
# Plot actual timeseries data
sns.lineplot(x="ds", y='y', data=df, color='green', label='Actual Time Series')
# Plot forecasted values
sns.lineplot(x="ds", y='timesfm', data=timesfm_forecast, color='red', label='Forecast')
plt.xlabel('Date')
plt.ylabel('Electric Production')
# Show the legend
plt.legend()
# Display the plot
plt.savefig('TimesFM_Electricity_Plot.png')
plt.show()

#calculationg the errors
import numpy as np
actuals = test_df['y'] # add .iloc[-60:] for the last 60 datapoints, and change for the 7 and 30 day forecast to [:7]/[:30], no need to run the entire code again, only the part which calculates the errors
predicted_values = timesfm_forecast['timesfm'] # same as above and always parallel
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


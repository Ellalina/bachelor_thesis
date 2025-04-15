#Execution recommended in jupyter notebook
import pandas as pd
from nixtla import NixtlaClient

# Get your API Key at dashboard.nixtla.io
# 1. Instantiate the NixtlaClient
nixtla_client = NixtlaClient(api_key = 'nixtla-tok-Xfsa5rQzdeWsiq3PICaZgJoF0VFnv5bL0vLzSPzpGVFb9cLMCbTXpuTigA8TcvaAFRDUNwbdDvYflm5m') # maybe change to own APT key

# 2. Read historic electricity demand data
data = pd.read_csv('total_consumption.csv')

data['Date']=pd.to_datetime(data['Date']) #convert to year-month-day

data = data.set_index('Date')

from sklearn.preprocessing import MinMaxScaler
# Normalize data, not needed for MAPE, and change all scaled_df to data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
scaled_df = pd.DataFrame(scaled_data, columns=['Value', 'temp'])
# Mape: change "scaled_df" to "data"
scaled_df['Date'] = pd.date_range(start='2015-07-06', periods=len(scaled_df), freq='D')
columns_order = ['Date'] + [col for col in scaled_df.columns if col != 'Date']
scaled_df = scaled_df[columns_order]
scaled_df = scaled_df[-544:]

#Let's Visualise the Datas
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore') # Settings the warnings to be ignored
sns.set(style="darkgrid")
plt.figure(figsize=(15, 6))
sns.lineplot(x="Date", y='Value', data=scaled_df, color='green')
plt.xlabel('Date')
plt.ylabel('Electricity Consumption')
plt.show()

# Mape: change "scaled_df" to "data"
df = pd.DataFrame({'unique_id':[1]*len(scaled_df),'ds': scaled_df["Date"], 
"y":scaled_df['Value']})

# Spliting training and test set
split_idx = int(len(df) * 0.766)
# Split the dataframe into train = historical data and test sets
train_df = df[:split_idx]
test_df = df[split_idx:]
print(train_df.shape, test_df.shape)

# Forecast with TimeGPT
fcst_df = nixtla_client.forecast(train_df, h=128, model='timegpt-1-long-horizon') # change to: "fcst_df = nixtla_client.forecast(train_df, h=7)" for the 7 day forecast

# Let's Visualise the Data
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')  # Setting the warnings to be ignored
# Set the style for seaborn
sns.set(style="darkgrid")
# Plot size
plt.figure(figsize=(15, 6))
# Plot actual timeseries data
sns.lineplot(x="ds", y='TimeGPT', data=fcst_df, color='red', label='Forecast')
# Plot forecasted values
sns.lineplot(x="Date", y='Value', data=scaled_df, color='green', label='Actual Time Series')
# Set plot title and labels
plt.title('Electric Production: Actual vs Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
# Show the legend
plt.legend()
# Display the plot
plt.savefig('For_TimeGPT_plot.png')
plt.show()

import numpy as np
actuals = test_df['y'] # add .iloc[-60:] for the last 60 datapoints, and change for the 7 and 30 day forecast to [:7]/[:30], no need to run the entire code again, only the part which calculates the errors
predicted_values = fcst_df['TimeGPT'] # same as above and always parallel
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

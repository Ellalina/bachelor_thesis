#Execution recommended in jupyter notebook
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# %%
net_df = pd.read_csv("total_consumption.csv", index_col="Date", parse_dates=True)
net_df

# %%
plt.figure(figsize=(16,8))
plt.plot(net_df.index, net_df['Value'], color='green', label = 'Electricity Consumption')
plt.xlabel('Date')
plt.ylabel('Electricity Consumption')
plt.legend()
plt.grid(True)
#plt.savefig('Electricity_Con.png')
plt.show()

# %%
plot_acf(net_df["Value"], lags=1826/4) # interpretation of q = 1, 1826 = # of 
plt.savefig('PACF_long.png')
plot_pacf(net_df["Value"], lags=1826/4) # interpretation of p = 1
plt.savefig('ACF_long.png')

# %%
plt.figure(figsize=(16,8))
plt.plot(net_df.index, net_df['temp'], color='blue', label = 'Electricity Consumption')
plt.xlabel('Date')
plt.ylabel('Electricity Consumption')
plt.legend()
plt.grid(True)
#plt.savefig('Temp.png')
plt.show()

# %%
plot_acf(net_df["temp"], lags=1826/4) # interpretation of q = 1, 1826 = # of 
plt.savefig('PACF_long.png')
plot_pacf(net_df["temp"], lags=1826/4) # interpretation of p = 1
plt.savefig('ACF_long.png')

# %%
fig, ax1 = plt.subplots(figsize=(16, 8))

# Plot the first dataset on the primary y-axis
ax1.plot(net_df.index, net_df['Value'], color='green', label='Electricity Consumption')
ax1.set_xlabel('Date')
ax1.set_ylabel('Electricity Consumption', color='green')
ax1.tick_params(axis='y', labelcolor='green')
ax1.legend(loc='upper left')
ax1.grid(True)

# Create a secondary y-axis
ax2 = ax1.twinx()
ax2.plot(net_df.index, net_df['temp'], color='blue', label='Temperature')
ax2.set_ylabel('Temperature', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')
ax2.legend(loc='upper right')

# Show the plot
plt.tight_layout()
plt.show()

# %%
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose the first variable ('Electricity_Consumption')
decompose_temp = seasonal_decompose(net_df['Value'], period=365)
residuals_temp = decompose_temp.resid.dropna()

# Decompose the second variable ('Temperature')
decompose_metric = seasonal_decompose(net_df['temp'], period=365)
residuals_metric = decompose_metric.resid.dropna()

# Ensure both residuals are aligned by index
residuals_temp, residuals_metric = residuals_temp.align(residuals_metric, join='inner')

# Calculate correlation between residuals
correlation = residuals_temp.corr(residuals_metric)
print(f"The correlation between residuals is: {correlation:.2f}")



#_______________________________________________________________________________
#New File: ARIMA
#Execution recommended in jupyter notebook
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

net_df = pd.read_csv("total_consumption.csv", index_col="Date", parse_dates=True)

from sklearn.preprocessing import MinMaxScaler
# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(net_df)
scaled_df = pd.DataFrame(scaled_data, columns=net_df.columns)
scaled_df['Date'] = pd.date_range(start='2015-01-01', periods=len(scaled_df), freq='D')
columns_order = ['Date'] + [col for col in scaled_df.columns if col != 'Date']
scaled_df = scaled_df[columns_order]

scaled_df = scaled_df.set_index('Date')

scaled_df2 = scaled_df.diff(365)
scaled_df2 = scaled_df2[365:]

from statsmodels.tsa.stattools import adfuller

#calculate the p values with the dicky-fuller test
result = adfuller(scaled_df2["Value"])
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")

result = adfuller(scaled_df2["temp"])
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")

#Print the ACF and PCF with the deseasonalised values
plot_acf(scaled_df2["Value"], lags=1826/4) # interpretation of q = 1, 1826 = # of 
plt.savefig('PACF_long.png')
plot_pacf(scaled_df2["Value"], lags=1826/4) # interpretation of p = 1
plt.savefig('ACF_long.png')

#splitting the data into training and test set
train_data, test_data = scaled_df2[0:int(len(scaled_df2)*0.913)], scaled_df2[int(len(scaled_df2)*0.913):]
train_arima = train_data['Value']
test_arima = test_data['Value']

#Settting up the forecast
history = [x for x in train_arima]
y = test_arima
# make first prediction
predictions = list()
model = ARIMA(history, order=(1,0,1))
model_fit = model.fit()
yhat = model_fit.forecast()[0]
predictions.append(yhat)
history.append(y[0])

# rolling forecasts
for i in range(1, len(y)):
    # predict
    model = ARIMA(history, order=(1,0,1))
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    # invert transformed prediction
    predictions.append(yhat)
    # observation
    obs = y[i]
    history.append(obs)

#visualise data
import matplotlib.pyplot as plt
plt.figure(figsize=(16,8))
plt.plot(scaled_df2.index, scaled_df2['Value'], color='green', label = 'Trainings Data')
plt.plot(test_data.index, y, color = 'green', label = 'Real Data')
plt.plot(test_data.index, predictions, color = 'red', label = 'Predicted Data')
plt.xlabel('Date')
plt.ylabel('Electricity Consumption')
plt.legend()
plt.grid(True)
plt.savefig('arima_model_norm.png')
plt.show()


# back differentiation inot data with seasonality
past_values = scaled_df['Value'].iloc[-len(predictions) - 365:-365].values
reverted_predictions = np.array(predictions) + past_values

# additional values for the second dimension of the scaler
reverted_predictions_full = np.zeros((len(reverted_predictions), scaled_data.shape[1]))
reverted_predictions_full[:, 0] = reverted_predictions  # Annahme: 'Value' ist die erste Spalte

# Denormalise
original_scale_predictions_full = scaler.inverse_transform(reverted_predictions_full)

# defining a dataframe with new values
original_scale_predictions = original_scale_predictions_full[:, 0]
predictions_df = pd.DataFrame(original_scale_predictions, columns=['Value'])
dates = test_data.index  
predictions_df['Date'] = dates[-len(original_scale_predictions):]  
predictions_df = predictions_df.set_index('Date')  


# report performance
import numpy as np

#calculating error metrics
y_subset = y[:7]# change to [:30] for 30 day forecast, [-60:] for 60 days, and leave out for 128 days
predictions_subset = predictions[:7]# change to [:30] for 30 day forecast, [-60:] for 60 days, and leave out for 128 days

y_dNorm = net_df['Value'][-128:]
y_dNorm = y_dNorm[:7] # change to [:30] for 30 day forecast, [-60:] for 60 days, and leave out for 128 days
predictions_dNorm = predictions_df['Value']
predictions_dNorm = predictions_dNorm[:7]# change to [:30] for 30 day forecast, [-60:] for 60 days, and leave out for 128 days

mae = mean_absolute_error(y_subset, predictions_subset)
print('MAE: ' + str(mae))

mse = mean_squared_error(y_subset, predictions_subset)
print('MSE: ' + str(mse))

rmse = math.sqrt(mse)
print('RMSE: ' + str(rmse))

mape = np.mean(np.abs((y_dNorm - predictions_dNorm) / y_dNorm))  
print('MAPE: ' + str(mape))



# Do the Forecast with covariates
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Assume 'train_data' and 'test_data' include both the target variable ('Value') and covariates ('cov1', 'cov2').
train_values = train_data['Value']
test_values = test_data['Value']

# Prepare covariates for training and testing
train_covariates = train_data[['temp']]  # Select relevant covariates
test_covariates = test_data[['temp']]

# Initialize history and predictions
history_values = [x for x in train_values]
history_covariates = train_covariates.values.tolist()

predictions2 = []

# Rolling Forecast
for t in range(len(test_values)):
    # Train the SARIMAX model with the covariates
    model = SARIMAX(
        endog=history_values,
        exog=history_covariates,
        order=(1, 0, 1)  # Adjust ARIMA parameters as needed
    )
    model_fit = model.fit(disp=False)

    # Forecast the next value using the next covariate values
    next_covariates = test_covariates.iloc[t].values.reshape(1, -1)  # Ensure 2D shape for prediction
    yhat = model_fit.forecast(steps=1, exog=next_covariates)[0]
    predictions2.append(yhat)

    # Update history with the actual observed value and covariates
    history_values.append(test_values.iloc[t])
    history_covariates.append(next_covariates[0])

#visualise data
plt.figure(figsize=(16,8))
plt.plot(scaled_df2.index, scaled_df2['Value'], color='green', label = 'Trainings Data')
plt.plot(test_data.index, y, color = 'green', label = 'Real Data')
plt.plot(test_data.index, predictions2, color = 'red', label = 'Predicted Data')
plt.xlabel('Date')
plt.ylabel('Electricity Consumption')
plt.legend()
plt.grid(True)
plt.savefig('arima_model_cov_norm.png')
plt.show()

# back differentiation to forecast with seasonality
past_values = scaled_df['Value'].iloc[-len(predictions2) - 365:-365].values
reverted_predictions = np.array(predictions2) + past_values

# additional values for the second dimension of the scaler
reverted_predictions_full = np.zeros((len(reverted_predictions), scaled_data.shape[1]))
reverted_predictions_full[:, 0] = reverted_predictions  # Annahme: 'Value' ist die erste Spalte

# Denormalise
original_scale_predictions_full = scaler.inverse_transform(reverted_predictions_full)

# defining a dataframe with new values
original_scale_predictions = original_scale_predictions_full[:, 0]
predictions_df = pd.DataFrame(original_scale_predictions, columns=['Value'])
dates = test_data.index  
predictions_df['Date'] = dates[-len(original_scale_predictions):]  
predictions_df = predictions_df.set_index('Date')  

#visualise data
plt.figure(figsize=(16,8))
plt.plot(net_df.index, net_df['Value'], color='green', label = 'Trainings Data')
plt.plot(predictions_df.index, predictions_df['Value'], color = 'red', label = 'Predicted Data')
plt.xlabel('Date')
plt.ylabel('Electricity Consumption')
plt.legend()
plt.grid(True)
plt.savefig('Cov_arima_model.png')
plt.show()

#calculationg error metrics
y_subset = y[:7]# change to [:30] for 30 day forecast, [-60:] for 60 days, and leave out for 128 days
predictions_subset = predictions2[:7]# change to [:30] for 30 day forecast, [-60:] for 60 days, and leave out for 128 days

y_dNorm = net_df['Value'][-128:]
y_dNorm = y_dNorm[:7]# change to [:30] for 30 day forecast, [-60:] for 60 days, and leave out for 128 days
predictions_dNorm = predictions_df['Value']
predictions_dNorm = predictions_dNorm[:7]# change to [:30] for 30 day forecast, [-60:] for 60 days, and leave out for 128 days

mae = mean_absolute_error(y_subset, predictions_subset)
print('MAE: ' + str(mae))

mse = mean_squared_error(y_subset, predictions_subset)
print('MSE: ' + str(mse))

rmse = math.sqrt(mse)
print('RMSE: ' + str(rmse))

mape = np.mean(np.abs((y_dNorm - predictions_dNorm) / y_dNorm))  
print('MAPE: ' + str(mape))




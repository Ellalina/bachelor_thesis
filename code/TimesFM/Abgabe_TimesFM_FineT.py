#Execution recommended in jupyter notebook
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PMAP_USE_TENSORSTORE'] = 'false'

!pip install timesfm[pax] # only istall pax not torch

import timesfm

# set up model
tfm = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend="gpu", #Must be adjusted if necessary
          per_core_batch_size=32,
          horizon_len=128,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          huggingface_repo_id="google/timesfm-1.0-200m"),
  )

import gc
import numpy as np
import pandas as pd
from timesfm import patched_decoder
from timesfm import data_loader

from tqdm import tqdm
import dataclasses
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt

#read data
data = pd.read_csv('/content/total_consumption.csv')

#not needed if normalisation is not wanted, like for MAPE
data = data.set_index('Date')
from sklearn.preprocessing import MinMaxScaler
# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
scaled_df = pd.DataFrame(scaled_data, columns=['Value', 'temp'])
scaled_df['Date'] = pd.date_range(start='2015-01-01', periods=len(scaled_df), freq='D')
columns_order = ['Date'] + [col for col in scaled_df.columns if col != 'Date']
scaled_df = scaled_df[columns_order]
scaled_df.to_csv('modified_table.csv', index=False)  # index=False to exclude row numbers

# set up the dataframe for the training, validation and testing sets
DATA_DICT = {
    "consumption": {
        "boundaries": [736, 1280, 1824],
        "data_path": "/content/total_consumption.csv", # for normalised data: /content/modified_table.csv
        "freq": "D",
    },
}

#set up the dataset
dataset = "consumption"
data_path = DATA_DICT[dataset]["data_path"]
freq = DATA_DICT[dataset]["freq"]
int_freq = timesfm.freq_map(freq)
boundaries = DATA_DICT[dataset]["boundaries"]

data_df = pd.read_csv(open(data_path, "r"))
data_df

ts_cols = [col for col in data_df.columns if col not in ["Date", "temp"]]
num_cov_cols = None # for covariate: ['temp']
cat_cov_cols = None

context_len = 416
pred_len = 128

num_ts = len(ts_cols)
batch_size = 32

# set up the data_loader for the model
dtl = data_loader.TimeSeriesdata(
      data_path=data_path,
      datetime_col="Date",
      num_cov_cols=num_cov_cols,
      cat_cov_cols=cat_cov_cols,
      ts_cols=np.array(ts_cols),
      train_range=[0, boundaries[0]],
      val_range=[boundaries[0], boundaries[1]],
      test_range=[boundaries[1], boundaries[2]],
      hist_len=context_len,
      pred_len=pred_len,
      batch_size=num_ts,
      freq=freq,
      normalize=False,
      epoch_len=None,
      holiday=False,
      permute=False,
  )

#sort batches
train_batches = dtl.tf_dataset(mode="train", shift=1).batch(batch_size)
val_batches = dtl.tf_dataset(mode="val", shift=pred_len)
test_batches = dtl.tf_dataset(mode="test", shift=pred_len)

for tbatch in tqdm(train_batches.as_numpy_iterator()):
    pass
print(tbatch[0].shape)

# shows prior mae loss before training, not necessary
mae_losses = []
for batch in tqdm(test_batches.as_numpy_iterator()):
    past = batch[0]
    actuals = batch[3]
    forecasts, _ = tfm.forecast(list(past), [0] * past.shape[0], normalize=False)
    forecasts = forecasts[:, 0 : actuals.shape[1]]
    mae_losses.append(np.abs(forecasts - actuals).mean())

print(f"MAE: {np.mean(mae_losses)}")

# set up everything for training
import jax
from jax import numpy as jnp
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis import base_model
from praxis import optimizers
from praxis import schedules
from praxis import base_hyperparams
from praxis import base_layer
from paxml import tasks_lib
from paxml import trainer_lib
from paxml import checkpoints
from paxml import learners
from paxml import partitioning
from paxml import checkpoint_types

# PAX shortcuts
NestedMap = py_utils.NestedMap
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams
InstantiableParams = py_utils.InstantiableParams
JTensor = pytypes.JTensor
NpTensor = pytypes.NpTensor
WeightedScalars = pytypes.WeightedScalars
instantiate = base_hyperparams.instantiate
LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]
AuxLossStruct = base_layer.AuxLossStruct

AUX_LOSS = base_layer.AUX_LOSS
template_field = base_layer.template_field

# Standard prng key names
PARAMS = base_layer.PARAMS
RANDOM = base_layer.RANDOM

key = jax.random.PRNGKey(seed=1234)

model = pax_fiddle.Config(
    patched_decoder.PatchedDecoderFinetuneModel,
    name='patched_decoder_finetune',
    core_layer_tpl=tfm.model_p,
)

@pax_fiddle.auto_config
def build_learner() -> learners.Learner:
  return pax_fiddle.Config(
      learners.Learner,
      name='learner',
      loss_name='avg_qloss',
      optimizer=optimizers.Adam(
          epsilon=1e-7,
          clip_threshold=1e2,
          learning_rate=1e-2, # change learning rate to 1e-2 to 5e-2 for covariate setting when calculating the error metics for the 7, 30 and 128 forecasting range, all other ranges have the learing rates that is set
          lr_schedule=pax_fiddle.Config(
              schedules.Cosine,
              initial_value=1e-3,
              final_value=1e-4,
              total_steps=3100,
          ),
          ema_decay=0.9999,
      ),
      # Linear probing i.e we hold the transformer layers fixed.
      bprop_variable_exclusion=['.*/stacked_transformer_layer/.*'],
  )

task_p = tasks_lib.SingleTask(
    name='ts-learn',
    model=model,
    train=tasks_lib.SingleTask.Train(
        learner=build_learner(),
    ),
)

task_p.model.ici_mesh_shape = [1, 1, 1]
task_p.model.mesh_axis_names = ['replica', 'data', 'mdl']

DEVICES = np.array(jax.devices()).reshape([1, 1, 1])
MESH = jax.sharding.Mesh(DEVICES, ['replica', 'data', 'mdl'])

num_devices = jax.local_device_count()
print(f'num_devices: {num_devices}')
print(f'device kind: {jax.local_devices()[0].device_kind}')

jax_task = task_p
key, init_key = jax.random.split(key)

# To correctly prepare a batch of data for model initialization (now that shape
# inference is merged), we take one devices*batch_size tensor tuple of data,
# slice out just one batch, then run the prepare_input_batch function over it.


def process_train_batch(batch):
    past_ts = batch[0].reshape(batch_size * num_ts, -1)
    actual_ts = batch[3].reshape(batch_size * num_ts, -1)
    return NestedMap(input_ts=past_ts, actual_ts=actual_ts)


def process_eval_batch(batch):
    past_ts = batch[0]
    actual_ts = batch[3]
    return NestedMap(input_ts=past_ts, actual_ts=actual_ts)


jax_model_states, _ = trainer_lib.initialize_model_state(
    jax_task,
    init_key,
    process_train_batch(tbatch),
    checkpoint_type=checkpoint_types.CheckpointType.GDA,
)

jax_model_states.mdl_vars['params']['core_layer'] = tfm._train_state.mdl_vars['params']
jax_vars = jax_model_states.mdl_vars
gc.collect()

jax_task = task_p


def train_step(states, prng_key, inputs):
  return trainer_lib.train_step_single_learner(
      jax_task, states, prng_key, inputs
  )


def eval_step(states, prng_key, inputs):
  states = states.to_eval_state()
  return trainer_lib.eval_step_single_learner(
      jax_task, states, prng_key, inputs
  )

key, train_key, eval_key = jax.random.split(key, 3)
train_prng_seed = jax.random.split(train_key, num=jax.local_device_count())
eval_prng_seed = jax.random.split(eval_key, num=jax.local_device_count())

p_train_step = jax.pmap(train_step, axis_name='batch')
p_eval_step = jax.pmap(eval_step, axis_name='batch')

replicated_jax_states = trainer_lib.replicate_model_state(jax_model_states)
replicated_jax_vars = replicated_jax_states.mdl_vars

best_eval_loss = 1e7
step_count = 0
patience = 0
NUM_EPOCHS = 500
PATIENCE = 5
TRAIN_STEPS_PER_EVAL = 1000
#train_state_unpadded_shape_dtype_struct
CHECKPOINT_DIR='./checkpoints'

def reshape_batch_for_pmap(batch, num_devices):
  def _reshape(input_tensor):
    bsize = input_tensor.shape[0]
    residual_shape = list(input_tensor.shape[1:])
    nbsize = bsize // num_devices
    return jnp.reshape(input_tensor, [num_devices, nbsize] + residual_shape)

  return jax.tree.map(_reshape, batch)

#start training loop
for epoch in range(NUM_EPOCHS):
    print(f"__________________Epoch: {epoch}__________________", flush=True)
    train_its = train_batches.as_numpy_iterator()
    if patience >= PATIENCE:
        print("Early stopping.", flush=True)
        break
    for batch in tqdm(train_its):
        train_losses = []
        if patience >= PATIENCE:
            print("Early stopping.", flush=True)
            break
        tbatch = process_train_batch(batch)
        tbatch = reshape_batch_for_pmap(tbatch, num_devices)
        replicated_jax_states, step_fun_out = p_train_step(
            replicated_jax_states, train_prng_seed, tbatch
        )
        train_losses.append(step_fun_out.loss[0])
        if step_count % TRAIN_STEPS_PER_EVAL == 0:
            print(
                f"Train loss at step {step_count}: {np.mean(train_losses)}",
                flush=True,
            )
            train_losses = []
            print("Starting eval.", flush=True)
            val_its = val_batches.as_numpy_iterator()
            eval_losses = []
            for ev_batch in tqdm(val_its):
                ebatch = process_eval_batch(ev_batch)
                ebatch = reshape_batch_for_pmap(ebatch, num_devices)
                _, step_fun_out = p_eval_step(
                    replicated_jax_states, eval_prng_seed, ebatch
                )
                eval_losses.append(step_fun_out.loss[0])
            mean_loss = np.mean(eval_losses)
            print(f"Eval loss at step {step_count}: {mean_loss}", flush=True)
            if mean_loss < best_eval_loss or np.isnan(mean_loss):
                best_eval_loss = mean_loss
                print("Saving checkpoint.")
                jax_state_for_saving = py_utils.maybe_unreplicate_for_fully_replicated(
                    replicated_jax_states
                )
                checkpoints.save_checkpoint(
                    jax_state_for_saving, CHECKPOINT_DIR, overwrite=True
                )
                patience = 0
                del jax_state_for_saving
                gc.collect()
            else:
                patience += 1
                print(f"patience: {patience}")
        step_count += 1

train_state = checkpoints.restore_checkpoint(jax_model_states, CHECKPOINT_DIR)

print(train_state.step)
tfm._train_state.mdl_vars['params'] = train_state.mdl_vars['params']['core_layer']
tfm.jit_decode()

#mae loss after training, not necessary for results
mae_losses = []
for batch in tqdm(test_batches.as_numpy_iterator()):
    past = batch[0]
    actuals = batch[3]
    _, forecasts = tfm.forecast(list(past), [0] * past.shape[0])
    forecasts = forecasts[:, 0 : actuals.shape[1], 5]
    mae_losses.append(np.abs(forecasts - actuals).mean())

print(f"MAE: {np.mean(mae_losses)}")

#setup for graph of the actual values and forecasted values
# Flatten all arrays inside cov_forecast into a single list
flattened_past = [item for sublist in past for item in sublist]
# Convert the flattened list into a pandas Series or DataFrame
flattened_train_df = pd.DataFrame(flattened_past, columns=["Value"])
flattened_train_df['Date'] = pd.date_range(start='2018-07-04', periods=len(flattened_train_df), freq='D')

# Flatten all arrays inside cov_forecast into a single list
flattened_actuals = [item for sublist in actuals for item in sublist]
# Convert the flattened list into a pandas Series or DataFrame
flattened_test_df = pd.DataFrame(flattened_actuals, columns=["Value"])
flattened_test_df['Date'] = pd.date_range(start='2019-08-24', periods=len(flattened_test_df), freq='D')

# Flatten all arrays inside cov_forecast into a single list
flattened_values = [item for sublist in forecasts for item in sublist]
# Convert the flattened list into a pandas Series or DataFrame
flattened_df = pd.DataFrame(flattened_values, columns=["Value"])
flattened_df['Date'] = pd.date_range(start='2019-08-24', periods=len(flattened_df), freq='D')

# Let's Visualise the Data
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')  # Setting the warnings to be ignored
# Set the style for seaborn
sns.set(style="darkgrid")
# Plot size
plt.figure(figsize=(15, 6))
# Plot Actual Time Series
sns.lineplot(x="Date", y='Value', data=flattened_train_df, color='green', label='Actual Time Series')
sns.lineplot(x="Date", y='Value', data=flattened_test_df, color='green', label='Actual Time Series')
# Plot forecasted values
sns.lineplot(x="Date", y='Value', data=flattened_df, color='red', label='Forecast')
plt.xlabel('Date')
plt.ylabel('Electric Production')
# Show the legend
plt.legend()
# Display the plot
plt.savefig('FineT_For_Cov_Electricity_plot.png')
plt.show()

#calcualting the losses
mse_losses = []
rmse_losses = []
mape_losses = []
r2_losses = []
mae_losses = []

for batch in tqdm(test_batches.as_numpy_iterator()): #in each batch is in this case only one
    past = batch[0]
    actuals = batch[3]

    # Forecast using the model
    _, forecasts = tfm.forecast(list(past), [0] * past.shape[0])
    forecasts = forecasts[:, 0 : actuals.shape[1], 5]  # Select appropriate forecast range

    #forecasts = forecasts[:, :30] # needs to be changed for each forecast so, 7 day forecast: [:, :7], 30 day forecast: [:, :30], last 60 day forecast:[:, -60:] and for the 128 day forecast they can be deleted
    #actuals = actuals[:, :30] # the same as above, no need to run the entire code again, only the part which calculates the errors
    
    # Calculate MAE
    mae_losses.append(np.abs(forecasts - actuals).mean())

    # Calculate MSE
    mse = ((forecasts - actuals) ** 2).mean()
    mse_losses.append(mse)

    # Calculate RMSE
    rmse = mse ** 0.5
    rmse_losses.append(rmse)

    # Calculate MAPE
    mape = (np.abs((actuals - forecasts) / actuals)).mean()
    mape_losses.append(mape)


# Print the aggregated metrics
print(f"MAE: {np.mean(mae_losses)}")
print(f"MSE: {np.mean(mse_losses)}")
print(f"RMSE: {np.mean(rmse_losses)}")
print(f"MAPE: {np.mean(mape_losses)}")
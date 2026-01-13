# Detailed MLP-EMOS model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import torch
import torchist
import torch.nn as nn
import os
from datetime import datetime, timedelta
# Mumbai	18.25	73.5
# Kerala	10   	76.75
# Shimla	32	    76.5
# Delhi  	28.5	77.25
# Hyderabad	17.5	78
# Patna 	26.5	85
# Bhubanes  20.5	85.75
# Meghalaya	25.5	91.5

LATITUDE, LONGITUDE = 25, 74.25

# Define file paths
FORC_FILEPATH = '/home/rossby/ROSSBY/regridded-NEPS-G-rainfall-forc'
FORC_FILENAME_PATTERN = 'regridded_ind-neps-day01-memrf-*.nc'
OBS_FILEPATHS = [
    '/home/rossby/ROSSBY/Observation/RF25_ind2018_rfp25.nc',
    '/home/rossby/ROSSBY/Observation/RF25_ind2019_rfp25.nc',
    '/home/rossby/ROSSBY/Observation/RF25_ind2020_rfp25.nc',
    '/home/rossby/ROSSBY/Observation/RF25_ind2021_rfp25.nc',
    '/home/rossby/ROSSBY/Observation/RF25_ind2022_rfp25.nc']

# Load weather regimes data if possible (The number of input variables will be 23 + 7 = 30)
# WEATHER_PATTERNS_FILE = '/home/rossby/ROSSBY/daily_weather_regimes_frequencies.csv'
# weather_patterns = pd.read_csv(WEATHER_PATTERNS_FILE, index_col=0)
# weather_patterns.index = pd.to_datetime(weather_patterns.index).normalize()
# weather_patterns = weather_patterns.applymap(lambda x: float(x.strip('%')) if isinstance(x, str) else x)

# Define the model structure
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(23, 23)
        # Add other layers if possible
        # self.fc2 = nn.Linear(23,64)
        # self.fc3 = nn.Linear(64,23)
        # Add Dropout layer if possible
        # self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return x
def crps_wrapper(t, obs):
    if obs.numel() == 0 or t.numel() == 0:
        raise ValueError("Observed data (obs) or forecast data (t) is empty.")
    if len(t) != len(obs):
        raise ValueError("Forecast and observed data lengths do not match.")
    crps = custom_crps(t[0], obs[0].reshape(1))
    for i in range(1, len(obs)):
        crps = torch.cat((crps, custom_crps(t[i], obs[i].reshape(1))))
    return torch.mean(crps)

def custom_crps(t, obs):
    if obs.numel() == 0:
        raise ValueError("Observed data (obs) is empty.")
    t = torch.sort(t)[0]
    bin_edges = torch.cat((t, (t[-1] + 1000).reshape(1)))
    pdf = torchist.histogramdd(t.unsqueeze(-1), edges=bin_edges)
    pdf = torchist.normalize(pdf)[0]
    cdf = torch.cumsum(pdf, dim=0)
    if obs not in t:
        bin_edges = torch.cat((bin_edges, obs))
        bin_edges = torch.sort(bin_edges)[0]
        index = (bin_edges == obs).nonzero(as_tuple=False)
        if index.numel() > 0:
            index = index.item()
            if index != 0:
                cdf = torch.cat((cdf, cdf[index - 1].reshape(1)))
            else:
                cdf = torch.cat((cdf, torch.tensor([0])))
        else:
            raise ValueError(f"Observed value {obs.item()} not found in bin_edges.")
    cdf = torch.sort(cdf)[0]
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    obs_h = torch.ge(bin_edges[:-1], obs).int()
    crps = torch.sum(torch.abs(cdf - obs_h) ** 2 * bin_widths, dim=0).reshape(1)
    return crps

def load_forc(start_date, end_date):
    days_no = (end_date - start_date).days + 1
    forc_tensor = torch.full((days_no, 23), float('nan'))
    dates_list = pd.date_range(start=start_date, end=end_date).strftime('%Y%m%d').tolist()

    for i, date in enumerate(dates_list):
        for j in range(1, 24):
            path = os.path.join(FORC_FILEPATH, date, FORC_FILENAME_PATTERN.replace('*', str(j)))
            try:
                ds = xr.open_dataset(path)
                df = ds.to_dataframe().reset_index()
                if 'APCP_surface' in df.columns:
                    row = df[(df['LATITUDE'] == LATITUDE) & (df['LONGITUDE'] == LONGITUDE)]
                    if not row.empty:
                        forc_tensor[i, j - 1] = torch.tensor(row['APCP_surface'].values[0])
                    else:
                        print(
                            f"Coordinates ({LATITUDE}, {LONGITUDE}) not found in DataFrame for date {date}, member {j}.")
                else:
                    print(f"'APCP_surface' column not found for date {date}, member {j}.")
            except FileNotFoundError:
                pass

    forc_tensor = forc_tensor[~torch.isnan(forc_tensor).any(dim=1)]
    forc_tensor = torch.sort(forc_tensor, dim=1)[0]
    return forc_tensor

# The function for loading ensemble forecasts with additional input weather regimes
# def load_forc_with_patterns(weather_patterns, start_date, end_date):
#     days_no = (end_date - start_date).days + 1
#     forc_tensor = torch.full((days_no, 30), float('nan'))
#     dates_list = pd.date_range(start=start_date, end=end_date).strftime('%Y%m%d').tolist()
#     for i, date in enumerate(dates_list):
#         for j in range(1, 24):
#             path = os.path.join(FORC_FILEPATH, date, FORC_FILENAME_PATTERN.replace('*', str(j)))
#             try:
#                 ds = xr.open_dataset(path)
#                 df = ds.to_dataframe().reset_index()
#                 if 'APCP_surface' in df.columns:
#                     row = df[(df['LATITUDE'] == LATITUDE) & (df['LONGITUDE'] == LONGITUDE)]
#                     if not row.empty:
#                         forc_tensor[i, j - 1] = torch.tensor(row['APCP_surface'].values[0])
#                     else:
#                         print(
#                             f"Coordinates ({LATITUDE}, {LONGITUDE}) not found in DataFrame for date {date}, member {j}.")
#                 else:
#                     print(f"'APCP_surface' column not found for date {date}, member {j}.")
#             except FileNotFoundError:
#                 pass
#         try:
#             forc_tensor[i, 23:] = torch.tensor(weather_patterns.loc[pd.to_datetime(date)].values)
#         except KeyError:
#             pass
#             forc_tensor[i, 23:] = torch.tensor([float('nan')] * 7)
#
#     forc_tensor = forc_tensor[~torch.isnan(forc_tensor).any(dim=1)]
#     forc_tensor = torch.sort(forc_tensor, dim=1)[0]
#     return forc_tensor

def load_obs(start_date, end_date, obs_filepaths):
    all_obs_tensors = []
    for filepath in obs_filepaths:
        try:
            imd = xr.open_dataset(filepath)
            df = imd.to_dataframe().reset_index()
            obs_data = df[(df['LATITUDE'] == LATITUDE) & (df['LONGITUDE'] == LONGITUDE) &
                          (df['TIME'] >= start_date) & (df['TIME'] <= end_date)]
            if not obs_data.empty:
                obs_tensor = torch.tensor(obs_data['RAINFALL'].values)
                obs_tensor = obs_tensor[~torch.isnan(obs_tensor)]
                all_obs_tensors.append(obs_tensor)
            else:
                pass
        except FileNotFoundError:
            pass
    if all_obs_tensors:
        return torch.cat(all_obs_tensors)
    else:
        return torch.tensor([])

# Parameters setup
years = [2018, 2019, 2020, 2021, 2022]
num_epochs = 2600
train_losses = []
val_losses = []
validation_results = []

# The Function for defining time range of training and validation datasets
def load_multiple_years(start_year, end_year, start_month, end_month, obs_filepaths):
    all_forecast_tensors = []
    all_observed_tensors = []

    for year in range(start_year, end_year + 1):
        start_date = datetime(year, start_month, 1)
        end_date = datetime(year, end_month, 30)
        rain_forecast = load_forc(start_date, end_date)
        rain_observed = load_obs(start_date, end_date, obs_filepaths)
        if rain_forecast.numel() == 0 or rain_observed.numel() == 0:
            print(f"Skipping year {year} due to missing data.")
            continue
        all_forecast_tensors.append(rain_forecast)
        all_observed_tensors.append(rain_observed)
    combined_forecast_tensor = torch.cat(all_forecast_tensors, dim=0)
    combined_observed_tensor = torch.cat(all_observed_tensors, dim=0)

    return combined_forecast_tensor, combined_observed_tensor

# 5-fold Cross-validation setup
for validation_year in years:
    train_years = [year for year in years if year != validation_year]

    # Load training data
    combined_forecast_train, combined_observed_train = load_multiple_years(train_years[0], train_years[-1], 6, 9,OBS_FILEPATHS)
    if combined_forecast_train.numel() == 0 or combined_observed_train.numel() == 0:
        raise ValueError("Combined forecast or observed training data is empty.")

    # Ensure the forecast and observed data lengths match
    min_length = min(len(combined_forecast_train), len(combined_observed_train))
    combined_forecast_train = combined_forecast_train[:min_length]
    combined_observed_train = combined_observed_train[:min_length]

    # Load validation data
    val_start_date = datetime(validation_year, 6, 1)
    val_end_date = datetime(validation_year, 9, 30)
    rain_forecast_val = load_forc(val_start_date, val_end_date)

    # Load forcasts with weather patterns if possible
    # rain_forecast_val = load_forc_with_patterns(weather_patterns, start_date=val_start_date, end_date=val_end_date)

    rain_observed_val = load_obs(val_start_date, val_end_date, OBS_FILEPATHS)

    if rain_forecast_val.numel() == 0 or rain_observed_val.numel() == 0:
        raise ValueError("Forecast or observed validation data is empty.")

    # Ensure the forecast and observed data lengths match
    min_length_val = min(len(rain_forecast_val), len(rain_observed_val))
    rain_forecast_val = rain_forecast_val[:min_length_val]
    rain_observed_val = rain_observed_val[:min_length_val]

    # Instantiate the MLP model for each fold
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Add L2 Regularization if possible
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    fold_train_losses = []
    fold_val_losses = []

    for epoch in range(num_epochs):
        model.train()
        forecast_train_pred = model(combined_forecast_train)
        if combined_forecast_train.numel() == 0 or combined_observed_train.numel() == 0:
            raise ValueError("Forecast or observed training data is empty.")
        loss = crps_wrapper(forecast_train_pred, combined_observed_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            fold_train_losses.append(loss.item())
            model.eval()
            with torch.no_grad():
                val_loss = crps_wrapper(model(rain_forecast_val), rain_observed_val)
                fold_val_losses.append(val_loss.item())
            print(f'Validation Year {validation_year}, Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')
    with torch.no_grad():
        val_forecast_pred = model(rain_forecast_val)
        for i in range(min_length_val):
            date = val_start_date + timedelta(days=i)
            observed_values = rain_observed_val[i].item()
            raw_forecasts = rain_forecast_val[i].detach().numpy()
            raw_crps = custom_crps(rain_forecast_val[i],rain_observed_val[i].reshape(1)).item()
            postprocessed_forecasts = val_forecast_pred[i].detach().numpy()
            postprocessed_crps = custom_crps(val_forecast_pred[i], rain_observed_val[i].reshape(1)).item()
            validation_results.append({
                "Date": date,
                "observed values":observed_values,
                "raw forecasts":raw_forecasts,
                "CRPS for raw forecasts":raw_forecasts,
                "postprocessed forecasts": postprocessed_forecasts,
                "CRPS for postprocessed forecasts": postprocessed_crps
            })

# Save the validation results and trained model for testing other datasets
results_df = pd.DataFrame(validation_results)
results_df.to_csv('/home/rossby/ROSSBY/validation_results_Ra.csv', index=False)
torch.save(model, '/home/rossby/ROSSBY/model_Rajasthan.pt')


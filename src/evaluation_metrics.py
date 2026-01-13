# Evaluation metrics such as local histogram, rank histogram, CRPSS and BS

## Local histogram including the rainfall frequency distribution

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

observed_data = pd.read_excel('D:\CodingPlace\CodingPlace\.venv\Data\IMD.xlsx')
raw_forecasts = pd.read_csv('/CodingPlace/.venv/extracted_raw_forecasts.csv')
postprocessed_forecasts = pd.read_csv('/CodingPlace/.venv/extracted_postprocessed_forecasts.csv')

combined_raw_forecasts = pd.concat([raw_forecasts[col] for col in raw_forecasts.columns])
combined_postprocessed_forecasts = pd.concat([postprocessed_forecasts[col] for col in postprocessed_forecasts.columns])

combined_raw_forecasts.dropna(inplace=True)
combined_postprocessed_forecasts.dropna(inplace=True)
observed_data.dropna(subset=['observed values'],inplace=True)

observed_data = pd.to_numeric(observed_data['observed values'],errors='coerce')
combined_raw_forecasts = pd.to_numeric(combined_raw_forecasts,errors='coerce')
combined_postprocessed_forecasts = pd.to_numeric(combined_postprocessed_forecasts,errors='coerce')

bins = [0,20,40,60,80,100,120,140,160,float('inf')]
labels = ['0-20','20-40','40-60','60-80','80-100','100-120','120-140','140-160','160-inf']

IMD_binned = pd.cut(observed_data,bins=bins,labels=labels).value_counts(normalize=True).sort_index()
raw_binned = pd.cut(combined_raw_forecasts,bins=bins,labels=labels).value_counts(normalize=True).sort_index()
postprocessed_binned = pd.cut(combined_postprocessed_forecasts,bins=bins,labels=labels).value_counts(normalize=True).sort_index()

fig = plt.figure(figsize=(8,4))
x = range(len(raw_binned.index))

ax1 = fig.add_subplot(1,1,1)
ax1.bar(x,IMD_binned,width=0.2,color='green',label='IMD',edgecolor='black')
ax1.bar([p+ 0.2 for p in x],raw_binned,width=0.2,color='orange',label='Raw Forecast',edgecolor='black')
ax1.bar([p+2* 0.2 for p in x],postprocessed_binned,width=0.2,color='pink',label='ML-EMOS',edgecolor='black')
ax1.set_title('Indian City',fontsize=14)
plt.xticks([p+1.5* 0.2 for p in x],raw_binned.index,rotation=25,fontsize=10)
plt.yticks(fontsize=10)
ax1.grid(True,which='both',linestyle='--',linewidth=0.5,alpha=0.7,color='gray')
ax1.legend()
plt.show()

## Rank histogtram to identify the performance of ensemble members

observed_data = observed_data.apply(pd.to_numeric,errors = 'coerce')
raw_forecasts = raw_forecasts.apply(pd.to_numeric,errors = 'coerce')
postprocessed_forecasts = postprocessed_forecasts.apply(pd.to_numeric,errors = 'coerce')

min_rows = min(observed_data.shape[0],raw_forecasts.shape[0],postprocessed_forecasts.shape[0])
observed_data = observed_data.iloc[:min_rows]
raw_forecasts = raw_forecasts.iloc[:min_rows]
postprocessed_forecasts = postprocessed_forecasts.iloc[:min_rows]

ranks_raw = []
ranks_postprocessed = []

for day in range(min_rows):
    raw_forecasts_sorted = raw_forecasts.iloc[day].sort_values()
    postprocessed_forecasts_sorted = postprocessed_forecasts.iloc[day].sort_values()
    observed_values = observed_data.iloc[day,0]

    rank_raw = np.searchsorted(raw_forecasts_sorted,observed_values,side='right')
    rank_postprocessed = np.searchsorted(postprocessed_forecasts_sorted,observed_values,side='right')

    ranks_raw.append(rank_raw)
    ranks_postprocessed.append(rank_postprocessed)

bins = np.arange(1,25)
hist_raw,_ = np.histogram(ranks_raw,bins=bins,density=True)
hist_postprocessed,_ = np.histogram(ranks_postprocessed,bins=bins,density=True)

fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(1,1,1)
ax1.bar(bins[:-1]-0.2,hist_raw,width=0.2,label='Raw Forecast',color='orange',edgecolor='black')
ax1.bar(bins[:-1],hist_postprocessed,width=0.2,label='ML-EMOS',color='cyan',edgecolor='black')
ax1.set_title('Indian City')
ax1.set_ylim(0,0.7)

ax1.legend()
ax1.grid(True,linestyle='--',linewidth=0.5,color='gray')
plt.show()

## CRPS Score to examine the overall accuracy of the ensemble forecasts; Brier Score has the similar calculation
## Brier Score to examine the accuracy of the extreme precipitaion events (BS doesn't include in this code and only need to calculate the 90th percentile to identify extreme events)

import matplotlib.pyplot as plt
import numpy as np

# Input mean CRPS value of nine locations
mean_crps_raw = np.array([1,2,3,4,5,6,7,8,9])
mean_crps_ML1 = np.array([0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69])
mean_crps_ML2 = np.array([0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59])

# Calculate CRPSS of each ML model (Same for BSS)
crpss_ML1 = 1 - mean_crps_ML1/mean_crps_raw
crpss_ML2 = 1 - mean_crps_ML2/mean_crps_raw

# Define 9 locations and the length of x
locations = ['Mumbai','Rajasthan','Kerala','Shimla','Delhi','Hyderabad','Patna','Bhubaneswar','Meghalaya']
x = np.arange(len(locations))

# Plotting
fig = plt.figure(figsize=(9,6))
ax1 = fig.add_subplot(1,1,1)
width = 0.35
bars1 = ax1.bar(x - width/2, crpss_ML1, width=width,label='ML1 model',color='yellow')
bars2 = ax1.bar(x + width/2, crpss_ML2, width=width,label='ML2 model',color='pink')
ax1.set_ylabel('CRPSS Values',fontsize = 15)
ax1.set_title('Comparison of CRPSS between ML1 model and ML2 model')
ax1.set_xticks(x,locations,rotation=20)
ax1.set_xticklabels(locations)
ax1.legend()

# Annotate crpss value for each bar
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

ax1.set_ylim(0, 1)
ax1.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
plt.tight_layout()
plt.show()



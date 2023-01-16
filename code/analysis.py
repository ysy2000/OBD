import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

data = pd.read_excel('/kaggle/input/obd-dataset/TestData.xlsx')
data.head()

print(data[data['TROUBLE_CODES'].notnull()].shape,data.shape)

data['TROUBLE_CODES'].unique() # check the DTC (diagnostic trouble codes) codes present in the data and verify if our required codes are present
# we could observe that null values are present in the data

data = data.dropna(subset = ['TROUBLE_CODES']).reset_index(drop=True) # getting rid of null values from trouble codes
data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'], unit='ms') # converting Timestamp to proper format.

data.loc[(data['TROUBLE_CODES'].str.contains('P007E')) | (data['TROUBLE_CODES'].str.contains('P007F'))].info()
# checking data of interest for null values and data sparsity
# we could see some of the columns like FUEL_LEVEL, BAROMETRIC_PRESSURE(KPA)..etc having no data. 

data.loc[(data['TROUBLE_CODES'].str.contains('P007E')) | (data['TROUBLE_CODES'].str.contains('P007F'))]

data.loc[(data['TROUBLE_CODES'].str.contains('P007E')) | (data['TROUBLE_CODES'].str.contains('P007F'))].nunique()
# checking data of interest for unique & constant values
# we could see some of the columns like MAKE, MODEL, DTC_NUMBER having only single value for all the timestamps.

req_data = data[['TIMESTAMP','TROUBLE_CODES','ENGINE_COOLANT_TEMP','ENGINE_LOAD','ENGINE_RPM','INTAKE_MANIFOLD_PRESSURE','AIR_INTAKE_TEMP','SPEED','SHORT TERM FUEL TRIM BANK 1','THROTTLE_POS','TIMING_ADVANCE']]
# from the data sanity checks we have selected the columns in the dataset which vary with time to play with. 
req_data.head()

req_data.iloc[:,1:].describe().round(2)
# checking the distributions of the sensor data - insights are given at the end of the script

req_data = req_data.interpolate()
# interpolating missing values in the sensor data

req_data.iloc[:,1:].describe().round(2)
# checking the distributions of the sensor data after interpolation - insights are given at the end of the script

plt.figure(figsize = (15,5))
sns.heatmap(req_data.iloc[:,1:].corr(),annot=True)
# checking the correlations between the various sensor data - currently pearson correlation alone is taken into picture

req_data_1 = req_data[req_data.TROUBLE_CODES.str.contains('P007E')].reset_index(drop=True)
req_data_2 = req_data[req_data.TROUBLE_CODES.str.contains('P007F')].reset_index(drop=True)
# data slicing based on Trouble codes

print(req_data.shape) # (11925, 11)
print(req_data_1.shape,req_data_2.shape)  # ((47, 11), (11, 11))

req_data_1 = req_data_1.set_index(req_data_1['TIMESTAMP']).resample('D').mean().reset_index().fillna(0)
req_data_1['TROUBLE_CODES'] = 'P007E'
req_data_2 = req_data_2.set_index(req_data_2['TIMESTAMP']).resample('D').mean().reset_index().fillna(0)
req_data_2['TROUBLE_CODES'] = 'P007F'
# generalizing trouble code column to two groups - P007E & P007F
combined_data_resampled = pd.concat([req_data_1,req_data_2],ignore_index=True,sort=True)

# retaining original data for comparision with the resampled data
data1 = req_data[req_data.TROUBLE_CODES.str.contains('P007E')].reset_index(drop=True)
data2 = req_data[req_data.TROUBLE_CODES.str.contains('P007F')].reset_index(drop=True)
# data slicing based on Trouble codes
combined_data = pd.concat([data1,data2],ignore_index=True,sort=True)

X = combined_data.drop(columns={'TROUBLE_CODES','TIMESTAMP'})
y = combined_data['TROUBLE_CODES']
X_resampled = combined_data_resampled.drop(columns={'TROUBLE_CODES','TIMESTAMP'})
y_resampled = combined_data_resampled['TROUBLE_CODES']

# VISUALIZATION
plt.figure(figsize = (15,5))
plt.plot(data1['TIMESTAMP'],data1['ENGINE_COOLANT_TEMP'], marker = "s", label = 'P007E')
plt.plot(data2['TIMESTAMP'],data2['ENGINE_COOLANT_TEMP'], marker = "o", label = 'P007F')
plt.legend(loc='best')
plt.ylabel('ENGINE_COOLANT_TEMP')
plt.show()

#For P007F DTC code, coolant temperature is high as the engine heats up due to the failure of bank 2.
# Dataset couldn't be downsampled as the trend of the data would drastically change for the given trouble codes.

data1['code'] = 1 # P007E
data2['code'] = 2 # P007F

plt.figure(figsize = (15,5))
plt.plot(data1['TIMESTAMP'],data1['code'], marker = "s", label = 'P007E')
plt.plot(data2['TIMESTAMP'],data2['code'], marker = "o", label = 'P007F')
plt.legend(loc='best')
plt.ylabel('code')
plt.show()

#For P007F DTC code, coolant temperature is high as the engine heats up due to the failure of bank 2.
# Dataset couldn't be downsampled as the trend of the data would drastically change for the given trouble codes.

mdl1 = LogisticRegression(random_state=0,solver='lbfgs',multi_class='auto').fit(np.array(X['ENGINE_COOLANT_TEMP']).reshape(-1,1) , y)
mdl1.score(np.array(X['ENGINE_COOLANT_TEMP']).reshape(-1,1),y)

# checking the accuracy of the classifier to decide on the relationship between sensor data and trouble code's
# accuracy score is low to set up a relationship

mdl1.predict(np.array([114.1]).reshape(-1,1)),mdl1.predict(np.array([114.2]).reshape(-1,1))
# from the classifier model decision threshold of sensor data between the fault codes = 114.1

plt.figure(figsize = (15,5))
plt.plot(req_data_1['TIMESTAMP'],req_data_1['ENGINE_COOLANT_TEMP'].rolling(window=3).mean(), marker = "s", label = 'P007E')
plt.plot(req_data_2['TIMESTAMP'],req_data_2['ENGINE_COOLANT_TEMP'].rolling(window=3).mean(), marker = "o", label = 'P007F')
plt.legend(loc='best')
plt.ylabel('ENGINE_COOLANT_TEMP')
plt.show()
# after downsampling
# P007E - sensor value has a dip from sept 1st to sept 4th, and increases after september 9th. 
# P007F - sensor value decreases to 0 until sept 3rd, and follows a seasonal pattern afterwards.

plt.figure(figsize = (15,5))
plt.plot(data1['TIMESTAMP'],data1['ENGINE_LOAD'], marker = "s", label = 'P007E')
plt.plot(data2['TIMESTAMP'],data2['ENGINE_LOAD'], marker = "o", label = 'P007F')
plt.legend(loc='best')
plt.ylabel('ENGINE_LOAD')
plt.show()

# Engine Loads are high for P007E codes in the initial timestamps.

mdl2 = LogisticRegression(random_state=0,solver='lbfgs',multi_class='auto').fit(np.array(X['ENGINE_LOAD']).reshape(-1,1) , y)
mdl2.score(np.array(X['ENGINE_LOAD']).reshape(-1,1),y)

# checking the accuracy of the classifier to decide on the relationship between sensor data and trouble code's
# accuracy score is low to set up a relationship

plt.figure(figsize = (15,5))
plt.plot(req_data_1['TIMESTAMP'],req_data_1['ENGINE_LOAD'].rolling(window=3).mean(), marker = "s", label = 'P007E')
plt.plot(req_data_2['TIMESTAMP'],req_data_2['ENGINE_LOAD'].rolling(window=3).mean(), marker = "o", label = 'P007F')
plt.legend(loc='best')
plt.ylabel('ENGINE_LOAD')
plt.show()
# after downsampling
# P007E - sensor value has a dip from sept 1st to sept 4th, and increases after september 9th. 
# P007F - sensor value decreases to 0 until sept 3rd, and follows a seasonal pattern afterwards.

plt.figure(figsize = (15,5))
plt.plot(data1['TIMESTAMP'],data1['ENGINE_RPM'], marker = "s", label = 'P007E')
plt.plot(data2['TIMESTAMP'],data2['ENGINE_RPM'], marker = "o", label = 'P007F')
plt.legend(loc='best')
plt.ylabel('ENGINE_RPM')
plt.show()

# ENGINE_RPM doesn't follow a specific trend for both P007E and P007F codes.mdl3 = LogisticRegression(random_state=0,solver='lbfgs',multi_class='auto').fit(np.array(X['ENGINE_RPM']).reshape(-1,1) , y)
mdl3.score(np.array(X['ENGINE_RPM']).reshape(-1,1),y)

# checking the accuracy of the classifier to decide on the relationship between sensor data and trouble code's
# accuracy score is low to set up a relationship
## 0.3620689655172414

plt.figure(figsize = (15,5))
plt.plot(req_data_1['TIMESTAMP'],req_data_1['ENGINE_RPM'].rolling(window=3).mean(), marker = "s", label = 'P007E')
plt.plot(req_data_2['TIMESTAMP'],req_data_2['ENGINE_RPM'].rolling(window=3).mean(), marker = "o", label = 'P007F')
plt.legend(loc='best')
plt.ylabel('ENGINE_RPM')
plt.show()
# after downsampling
# P007E - sensor value has a dip from sept 1st to sept 4th, and increases after september 9th. 
# P007F - sensor value decreases to 0 until sept 3rd, and follows a seasonal pattern afterwards.

plt.figure(figsize = (15,5))
plt.plot(data1['TIMESTAMP'],data1['INTAKE_MANIFOLD_PRESSURE'], marker = "s", label = 'P007E')
plt.plot(data2['TIMESTAMP'],data2['INTAKE_MANIFOLD_PRESSURE'], marker = "o", label = 'P007F')
plt.legend(loc='best')
plt.ylabel('INTAKE_MANIFOLD_PRESSURE')
plt.show()

#Air pressure is high for both the codes in initial timestamps

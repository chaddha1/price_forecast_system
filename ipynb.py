#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams 
rcParams["figure.figsize"]=10,6


# In[2]:


#parse srtings to datetime type 
dataset= pd.read_csv("datasets\dataset\potatoes.csv")
dataset["Month"]=pd.to_datetime(dataset["Month"],infer_datetime_format=True)
indexedDataset=dataset.set_index(["Month"])
#indexedDataset=lambda dates:pd.datetime.strptime(dates,"%Y-%m-%d")
#dataset=pd.read_csv("datasets\dataset\potatoes.csv",parse_dates=["Month"],index_col="Month",date_parser=indexedDataset)
#print(dataset.index)


# In[3]:


from datetime import datetime
indexedDataset.head(5)
#indexedDataset.tail(5)


# In[30]:


#plot graph
plt.xlabel('Date')
plt.ylabel("Price of Potatoes")
plt.plot(indexedDataset)


# In[31]:


#determining rolling statistics 
rolmean=indexedDataset.rolling(window=30).mean()
rolstd=indexedDataset.rolling(window=30).std()
print(rolmean,rolstd)


# In[32]:


#plot rolling statistics
orig=plt.plot(indexedDataset,color="blue",label="original")
mean=plt.plot(rolmean,color="red",label="rolling mean")
std=plt.plot(rolstd,color="black",label="rolling std")
plt.legend(loc="best")
plt.title("Rolling Mean and Standard Deviation")
plt.show(block=False)


# In[33]:


#perform dickey fuller test
from statsmodels.tsa.stattools import adfuller
print("Result of Dickey Fuller Test")
dftest=adfuller(indexedDataset['#Price'],autolag="AIC")
dfoutput=pd.Series(dftest[0:4],index=["test statistics", "p-value", "#lags used", "number of observations used"])
for key,value in dftest[4].items():
    dfoutput["Critical value(%s) "%key]=value
    
print(dfoutput)    


# In[34]:


#Estimating trend
indexedDataset_logScale=np.log(indexedDataset)
plt.plot(indexedDataset_logScale)


# In[36]:


movingAverage=indexedDataset_logScale.rolling(window=30).mean()
movingStd=indexedDataset_logScale.rolling(window=30).std()
plt.plot(indexedDataset_logScale)
plt.plot(movingAverage,color="red")


# In[37]:


datasetLogScaleMinusMovingAverage=indexedDataset_logScale-movingAverage
datasetLogScaleMinusMovingAverage.head(12)
#Remove NAN Vlaues
datasetLogScaleMinusMovingAverage.dropna(inplace=True)
datasetLogScaleMinusMovingAverage.head(10)


# In[38]:


from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    #determining rolling statistics
    movingAverage=timeseries.rolling(window=30).mean()
    movingstd=timeseries.rolling(window=30).std()
    #plot rolling statistics
    orig=plt.plot(timeseries,color="blue",label="original")
    mean=plt.plot(movingAverage,color="red",label="Rolling Mean")
    std=plt.plot(movingstd,color="black",label="Rolling Std")
    plt.legend(loc="best")
    plt.title("Rolling Mean And Standard Deviation")
    plt.show(block=False)
    
    #perform Dickey fuller test
    print("Results of Dickey fuller test")
    dftest=adfuller(timeseries["#Price"],autolag="AIC")
    dfoutput=pd.Series(dftest[0:4],index=["test Statistics", "p-value", "#Lags used", "Number of observations used"])
    for key,value in dftest[4].items():
        dfoutput["Critical value (%s)"%key]=value
    print(dfoutput)
    


# In[39]:


test_stationarity(datasetLogScaleMinusMovingAverage)


# In[40]:


exponentialDecayWeightedAverage=indexedDataset_logScale.ewm(halflife=30,min_periods=0,adjust=True).mean()
plt.plot(indexedDataset_logScale)
plt.plot(exponentialDecayWeightedAverage,color="red")


# In[41]:


datasetLogScaleMinusMovingExponentialDecayAverage=indexedDataset_logScale-exponentialDecayWeightedAverage
test_stationarity(datasetLogScaleMinusMovingExponentialDecayAverage)


# In[42]:


datasetLogDiffShifting=indexedDataset_logScale - indexedDataset_logScale.shift()
plt.plot(datasetLogDiffShifting)


# In[43]:


datasetLogDiffShifting.dropna(inplace=True)
test_stationarity(datasetLogDiffShifting)


# In[44]:


from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(indexedDataset_logScale)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(indexedDataset_logScale, label="Original")
plt.legend(loc="best")
plt.subplot(412)
plt.plot(trend, label="Trend")
plt.legend(loc="best")
plt.subplot(413)
plt.plot(seasonal, label="Seasonality")
plt.legend(loc="best")
plt.subplot(414)
plt.plot(residual, label="Residuals")
plt.legend(loc="best")
plt.tight_layout()

decomposedLogData=residual
decomposedLogData.dropna(inplace=True)
test_stationarity(decomposedLogData)


# In[45]:


decomposedLogData=residual
decomposedLogData.dropna(inplace=True)
test_stationarity(decomposedLogData)


# In[46]:


#ACF and PACF plots:
from statsmodels.tsa.stattools import acf,pacf

lag_acf=acf(datasetLogDiffShifting, nlags=20)
lag_pacf=pacf(datasetLogDiffShifting, nlags=20, method="ols")

#Pot ACF
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle="--",color="gray")
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle="--",color="gray")
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle="--",color="gray")
plt.title("Autocorrelation Function")

#plot PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle="--",color="gray")
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle="--",color="gray")
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle="--",color="gray")
plt.title("Partial Autocorrelation Function")
plt.tight_layout()


# In[47]:


from statsmodels.tsa.arima_model import ARIMA

#AR model
model=ARIMA(indexedDataset_logScale,order=(2,1,1))
results_AR=model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_AR.fittedvalues,color="red")
plt.title("RSS: %.4f"%sum((results_AR.fittedvalues-datasetLogDiffShifting["#Price"])**2))
print('Plotting AR Model')


# In[48]:


#MA Model
model = ARIMA(indexedDataset_logScale, order=(0,1,2))
results_MA=model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_MA.fittedvalues,color="red")
plt.title("RSS: %.4f"%sum((results_MA.fittedvalues-datasetLogDiffShifting["#Price"])**2))
print('Plotting MA Model')


# In[49]:


model=ARIMA(indexedDataset_logScale,order=(2,1,1))
results_ARIMA=model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_ARIMA.fittedvalues,color="red")
plt.title("RSS %.4f"% sum((results_ARIMA.fittedvalues-datasetLogDiffShifting["#Price"])**2))


# In[50]:


predictions_ARIMA_diff=pd.Series(results_ARIMA.fittedvalues,copy=True)
print(predictions_ARIMA_diff.head())


# In[51]:


#convert to cumulation sum
predictions_ARIMA_diff_cumsum=predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())


# In[52]:


predictions_ARIMA_log=pd.Series(indexedDataset_logScale['#Price'].iloc[0],index=indexedDataset_logScale.index)
predictions_ARIMA_log=predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
predictions_ARIMA_log.head()


# In[53]:


predictions_ARIMA=np.exp(predictions_ARIMA_log)
plt.plot(indexedDataset)
plt.plot(predictions_ARIMA)


# In[27]:


indexedDataset_logScale


# In[ ]:





# In[28]:


y=results_ARIMA.plot_predict(1,1095)
x=results_ARIMA.forecast(steps=365)


# In[29]:


x







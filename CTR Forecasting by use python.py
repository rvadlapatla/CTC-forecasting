#!/usr/bin/env python
# coding: utf-8

# In[18]:




pip install matplotlib statsmodels


# In[28]:


pip install statsmodels==0.13.0


# In[2]:


pip install --upgrade statsmodels


# In[29]:


pip install statsmodels


# In[33]:


pip install scipy==1.6.3


# In[1]:


import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt


# In[31]:


pip install scipy


# In[2]:


data =pd.read_csv("Downloads\ctr.csv")
data.head()


# In[3]:


# data preparation
data['Date'] =pd.to_datetime(data['Date'],format ='%Y/%m/%d')
data.set_index('Date',inplace=True)


# In[4]:


# visualize clicks and impressions
fig =go.Figure()
fig.add_trace(go.Scatter(x=data.index,y=data['Clicks'],mode ='lines',name ="clicks"))
fig.add_trace(go.Scatter(x= data.index,y=data["Impressions"],mode ="lines",name ="impression"))
fig.update_layout(title ="click and impression")
fig.show()


# In[5]:


# create a scatter plot to visualize the relationship between clicks and impression
fig = px.scatter(data,x='Clicks',y='Impressions',title="relationship between clic and impression",
                 labels={'Clicks':'Clicks','Impressions':'Impressions'})
fig.update_layout(xaxis_title='Clicks',yaxis_title ='Impressions')
fig.show()


# In[7]:


data["CTR"] =(data['Clicks']/data['Impressions'])*100
print(data["CTR"])
fig =px.line(data,x=data.index,y="CTR",title ='click through rate')
fig.show()


# In[8]:


data['Dayofweek'] =data.index.dayofweek
data['weekofmaonth']=data.index.week  //4
print(data.columns)


#eda based on dayofweek
day =data.groupby("Dayofweek")['CTR'].mean().reset_index()
day['dayofweek'] =['m','T','W','TH','F','S','SU']
fig =px.bar(day,x="dayofweek",y='CTR',title="Average ctr by day of the week")
fig.show()


# In[9]:


# create a new column 'daycategory'to categorize weeks day
data['daycategory'] =data['Dayofweek'].apply(lambda x:"weekend" if x >= 5 else 'weekday')

#calculate average ctr for weekdays and weekends

cal =data.groupby('daycategory')['CTR'].mean().reset_index()
fig =px.bar(cal, x='daycategory',y="CTR",title='comparing of ctr on weekdays and weekends ',labels ={"CTR": 'Avg CTR'})
fig.update_layout(yaxis_title='avg')
fig.show()


# In[10]:


# Group the data by 'DayCategory' and calculate the sum of Clicks and Impressions for each category
grope =data.groupby('daycategory')[['Clicks','Impressions']].sum().reset_index()
# create bar chart to visualize
fig = px.bar(grope,x="daycategory",y=['Clicks','Impressions'],title ="impresion and clicks",color_discrete_sequence=['blue','green'])
fig.update_layout(yaxis_title="count")
fig.show()


# These lines import the necessary functions from the statsmodels library for time series analysis. Specifically, ARIMA is used for fitting an ARIMA model, and plot_acf and plot_pacf are used for plotting the autocorrelation function (ACF) and partial autocorrelation function (PACF).
# 
#  the .dropna() removes the NaN value that appears in the resulting series due to the differencing operation.
# 
#  This line creates a Matplotlib figure (fig) with two subplots arranged in a single row and two columns. The figsize parameter sets the size of the figure.
# 
# These lines plot the autocorrelation function (ACF) on the first subplot (axes[0]) and the partial autocorrelation function (PACF) on the second subplot (axes[1]). The autocorrelation functions provide insights into the correlation between a time series and its lagged values.

# In[15]:


data.reset_index(drop=True,inplace =True)
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

# resetting index 
time_series =data['CTR']

#differenceing
diff_series = time_series.diff().dropna()
fig,axes = plt.subplots(1,2,figsize =(12,4))
plot_acf(diff_series,ax =axes[0])
plot_pacf(diff_series,ax =axes[1])
plt.show()


# In[8]:


from statsmodels.tsa.statespace.sarimax import SARIMAX

p, d, q, s = 1, 1, 1, 12

model = SARIMAX(time_series, order=(p, d, q), seasonal_order=(p, d, q, s))
results = model.fit()
print(results.summary())


# In[5]:


# Predict future values
future_steps = 100
predictions = result.predict(len(time_series), len(time_series) + future_steps - 1)
print(predictions)


# In[37]:


# Create a DataFrame with the original data and predictions
forecast = pd.DataFrame({'Original': time_series, 'Predictions': predictions})

# Plot the original data and predictions
fig = go.Figure()

fig.add_trace(go.Scatter(x=forecast.index, y=forecast['Predictions'],
                         mode='lines', name='Predictions'))

fig.add_trace(go.Scatter(x=forecast.index, y=forecast['Original'],
                         mode='lines', name='Original Data'))

fig.update_layout(title='CTR Forecasting',
                  xaxis_title='Time Period',
                  yaxis_title='Impressions',
                  legend=dict(x=0.1, y=0.9),
                  showlegend=True)

fig.show()


# In[ ]:





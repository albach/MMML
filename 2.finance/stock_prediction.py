# %%
import pandas as pd
import pandas_datareader.data as web
from pandas import Series, DataFrame
from pandas.plotting import scatter_matrix
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
from math import ceil
import numpy as np

#%%
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split 

#%%
import matplotlib as mpl
mpl.rc('figure', figsize=(8, 7))
mpl.__version__
# %%

start = datetime.datetime(2018, 1, 1)
end = datetime.datetime(2019, 8, 31)
# %%
# EC: Eco Colombian Gas Company
df = web.DataReader("EC", 'yahoo', start, end)
print(df.tail())
# %%
# Rolling Mean (Moving Average)
close_px = df['Adj Close']
mavg = close_px.rolling(window=100).mean()
# mavg_50 = close_px.rolling(window=50).mean()
# mavg_25 = close_px.rolling(window=25).mean()
mavg_10 = close_px.rolling(window=10).mean()
#%%
# Adjusting the style of matplotlib
style.use('ggplot')

close_px.plot(label='EC')
mavg.plot(label='mavg')
mavg_10.plot(label='mavg 10')
plt.legend()
# plt.show()

# %%
# Risk
# rets = close_px / close_px.shift(1) - 1
# rets.plot(label='return')

#%%
dfreg = df.loc[:,['Adj Close','Volume']]
dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

# %%
# Data cleaning

# Drop missing value
dfreg.fillna(value=-99999, inplace=True)
# We want to separate 1 percent of the data to forecast
forecast_out = int(ceil(0.01 * len(dfreg)))
# Separating the label here, we want to predict the AdjClose
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'], 1))
# Scale the X so that everyone can have the same distribution for linear regression
X = scale(X) #scikit preprocessing package
# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
# Separate label and identify it as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]

# %%
# Data model_selection
# split 80% of the data to training set while 20% of the data to test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# %%
# Linear regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, y_train)
# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)

# Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, y_train)

# KNN Regression
clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit(X_train, y_train)

#%%
# Confidence scores

confidencereg = clfreg.score(X_test, y_test)
confidencepoly2 = clfpoly2.score(X_test,y_test)
confidencepoly3 = clfpoly3.score(X_test,y_test)
confidenceknn = clfknn.score(X_test, y_test)

print('Linear Regression: {}'.format(confidencereg))
print('Quadratic Regression 2: {}'.format(confidencepoly2))
print('Quadratic Regression 3: {}'.format(confidencepoly3))
print('KNN Regression: {}'.format(confidenceknn))

#%%
# =========== FORECASTS ==========
#Forecast Linear Regression
# forecast_set = clfreg.predict(X_lately)
# dfreg['Forecast Linear'] = np.nan

# #Forecast Quadratic Regression 2
# forecast_set = clfpoly2.predict(X_lately)
# dfreg['Forecast Quadratic 2'] = np.nan

# #Forecast Quadratic Regression 3
# forecast_set = clfpoly2.predict(X_lately)
# dfreg['Forecast Quadratic 3'] = np.nan

# #Forecast KNN Regression
forecast_set = clfknn.predict(X_lately)
dfreg['Forecast knn'] = np.nan

# %%
last_date = dfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)

for i in forecast_set:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]

dfreg['Adj Close'].tail(500).plot()
dfreg['Forecast knn'].tail(500).plot()
print(dfreg.tail())
plt.legend(loc=2)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
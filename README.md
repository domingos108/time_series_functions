# Residual Hybrid Systems 
In src.hybrid_systems methods there are training and test procedures for Residual Hybrid Systems

----------------------------------
Implemented Hybrid Systems 
that performs the linear combination:
* Linear Combination based on Zhang <span id="a1">[[1]](#f1)</span>, which is available on additive_hybrid_model method;
* Nonlinear Combination  based on Santos Jr et al. <span id="a2">[[2]](#f2)</span>, which is available on nolic_model method;

# Time Series Functions
The src.time_series_functions.py has methods that can help the development of time series forecast models.  

----------------------------------
## Create Windowing
Create a time lag of univariate series. 
Parameters:
* df:one column time series (pandas data frame);
* lag_size: size of the time lag.

Returns:
* Lagged time series (pandas data frame).

Usage:
~~~python
import pandas as pd
import time_series_functions as tsf
ts = pd.read_csv(time_series_path,sep=',',names = ['actual'],dtype='float64') # open univariate time series
ts_windowed = tsf.create_windowing(df=ts,lag_size=3)
~~~
----------------------------------

## Time series metrics calculation
Generate time series metrics.

The metrics are:
* MSE - Mean Square Error:
* RMSE - Root Mean Square Error :
* MAPE - Mean Absolute Percentage Error:
* SMAPE  - Symmetric Mean Absolute Percentage Error:
* MAE - Mean Absolute Error:
* theil - U of Theil Statistics:
* ARV - Average Relative Variance:
* IA - Index of Agreement:
* POCID - Prediction of Change in Direction: 

More details about the metrics implementation are available in <span id="a3">[[3]](#f3)</span>, <span id="a4">[[4]](#f4)</span>, and <span id="a5">[[5]](#f5)</span>.

Parameters:
* y_true: target value (numpy array);
* y_pred: predicted value (numpy array).

Returns:
* The dictionary of metrics 

Usage:
~~~python
import time_series_functions as tsf
gerenerate_metric_results(y_true, y_pred)
~~~ 
----------------------------------

## References
1. <span id="f1"></span> Zhang, G. Peter. "Time series forecasting using a hybrid ARIMA and neural network model." Neurocomputing 50 (2003): 159-175. [$\hookleftarrow$](#a1)

2. <span id="f2"></span> Domingos, S. de O., João FL de Oliveira, and Paulo SG de Mattos Neto. "An intelligent hybridization of ARIMA with machine learning models for time series forecasting." Knowledge-Based Systems 175 (2019): 72-86. [$\hookleftarrow$](#a2)

3. <span id="f3"></span> Silva, David A., et al. "Measurement of fitness function efficiency using data envelopment analysis." Expert Systems with Applications 41.16 (2014): 7147-7160. [$\hookleftarrow$](#a3)

4. <span id="f4"></span> de Mattos Neto, Paulo SG, George DC Cavalcanti, and Francisco Madeiro. "Nonlinear combination method of forecasters applied to PM time series." Pattern Recognition Letters 95 (2017): 65-72. [$\hookleftarrow$](#a4)

5. <span id="f5"></span> Silva, Eraylson G., et al. "Improving the accuracy of intelligent forecasting models using the Perturbation Theory." 2018 International Joint Conference on Neural Networks (IJCNN). IEEE, 2018. [$\hookleftarrow$](#a5)

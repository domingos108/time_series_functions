# Time Series Functions
The time_series_functions.py has methods that can help the development of time series forecast models.  

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

More details about the metrics implementation are available in <span id="a1">[[1]](#f1)</span>, <span id="a2">[[2]](#f2)</span>, and <span id="a3">[[3]](#f3)</span>.

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

1. <span id="f1"></span> Silva, David A., et al. "Measurement of fitness function efficiency using data envelopment analysis." Expert Systems with Applications 41.16 (2014): 7147-7160. [$\hookleftarrow$](#a1)

2. <span id="f2"></span> de Mattos Neto, Paulo SG, George DC Cavalcanti, and Francisco Madeiro. "Nonlinear combination method of forecasters applied to PM time series." Pattern Recognition Letters 95 (2017): 65-72. [$\hookleftarrow$](#a2)

3. <span id="f3"></span> Silva, Eraylson G., et al. "Improving the accuracy of intelligent forecasting models using the Perturbation Theory." 2018 International Joint Conference on Neural Networks (IJCNN). IEEE, 2018. [$\hookleftarrow$](#a3)

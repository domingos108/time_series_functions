import src.hybrid_systems as hs 
from sklearn.neural_network import MLPRegressor
import pandas as pd
import src.time_series_functions as tsf


def linear_combination():
    linear_model_path = './data/colorado_river_ARIMA_result_80.txt' 
    arima = pd.read_csv(linear_model_path,sep=';')

    real = arima['y'].values 
    predicted = arima['l'].values  
    time_window = 4 
    base_model = MLPRegressor() # the selected parameters must be specified
    test_size = 149
    val_size = 149
    results = hs.additive_hybrid_model(predicted, real, time_window, 
                            base_model,test_size, val_size)
    print("TEST METRICS")
    print(results['test_metrics'])
    tsf.save_result(results, './result_data/colorado_river_linear_combination_result')

def nonlinear_combination():
    linear_model_path = './data/colorado_river_ARIMA_result_80.txt'
    additive_hybrid_model_path = './result_data/colorado_river_linear_combination_result-1404201586894381.pkl'
    arima = pd.read_csv(linear_model_path,sep=';')

    real = arima['y'].values
    predicted = arima['l'].values    

    hybrid_arima = tsf.open_saved_result(additive_hybrid_model_path)
    error_forecaster_tw = len(predicted) - len(hybrid_arima['predicted_values']) 
    
    linear_forecast = predicted[error_forecaster_tw:]
    nonlinear_forecast = hybrid_arima['predicted_values']
    real = real[error_forecaster_tw:]

    time_window = 4
    base_model = MLPRegressor() # the selected parameters must be specified
    test_size = 149
    val_size = 149

    results = hs.nolic_model(linear_forecast, real, nonlinear_forecast,time_window, 
                base_model, test_size,val_size)

    print("TEST METRICS")
    print(results['test_metrics'])

if __name__ == "__main__":
    nonlinear_combination()
import pandas as pd
import numpy as np
import pickle as pkl
import datetime
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pyplot as plt

class result_options:
    test_result=0
    val_result=1
    train_result=2
    save_result=3

def create_windowing(df, lag_size):
    final_df = None
    for i in range(0, (lag_size + 1)):
        serie = df.shift(i)
        if (i == 0):
            serie.columns = ['actual']
        else:
            serie.columns = [str('lag' + str(i))]
        final_df = pd.concat([serie, final_df], axis=1)

    return final_df.dropna()


def open_data_set(arquivo):# open univariate data set
    return pd.read_csv(arquivo,sep=',',names = ['actual'])

def arima_model(df,p,d,q):
    from statsmodels.tsa.arima_model import ARIMA
    model = ARIMA(endog=df.as_matrix(), order=(p, d, q)).fit()
    return model.predict(0)


def mean_square_error(y_true, y_pred):
    y_true = np.asmatrix(y_true).reshape(-1)
    y_pred = np.asmatrix(y_pred).reshape(-1)

    return np.square(np.subtract(y_true, y_pred)).mean()

def root_mean_square_error(y_true, y_pred):

    return mean_square_error(y_true, y_pred)**0.5


def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if len(np.where(y_true == 0)[0]) > 0:
        return np.inf
    else:
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    return np.mean(np.abs((y_true - y_pred) / (( np.abs(y_true) + np.abs(y_pred) )/2) ))

def mean_absolute_error(y_true, y_pred):
    
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    return np.mean(np.abs(y_true - y_pred))


def u_theil(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    error_sup = np.square(np.subtract(y_true, y_pred)).sum()
    error_inf = np.square(np.subtract(y_pred[0:(len(y_pred) - 1)], y_pred[1:(len(y_pred))])).sum()

    return error_sup / error_inf


def average_relative_variance(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mean = np.mean(y_true)

    error_sup = np.square(np.subtract(y_true, y_pred)).sum()
    error_inf = np.square(np.subtract(y_pred, mean)).sum()

    return error_sup / error_inf


def index_agreement(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mean = np.mean(y_true)

    error_sup = np.square(np.abs(np.subtract(y_true, y_pred))).sum()

    error_inf = np.abs(np.subtract(y_pred, mean)) + np.abs(np.subtract(y_true, mean))
    error_inf = np.square(error_inf).sum()

    return 1 - (error_sup / error_inf)


def prediction_of_change_in_direction(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    true_sub = np.subtract(y_true[0:(len(y_true) - 1)], y_true[1:(len(y_true))])
    pred_sub = np.subtract(y_pred[0:(len(y_pred) - 1)], y_pred[1:(len(y_pred))])

    mult = true_sub * pred_sub
    result = 0
    for m in mult:
        if m > 0:
            result = result + 1

    return (100 * (result / len(y_true)))


def make_metrics_avaliation(y_true, y_pred, test_size, val_size,return_type,model_params, title):
    data_size = len(y_true)
    train_size = data_size - (val_size + test_size)

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    y_true_test = y_true[(data_size - test_size):data_size]
    y_pred_test = y_pred[(data_size - test_size):data_size]

    val_result = None
    if val_size>0:
        y_true_val = y_true[(train_size):(data_size - test_size)]
        y_pred_val = y_pred[(train_size):(data_size - test_size)]
        val_result = gerenerate_metric_results(y_true_val, y_pred_val)

    y_true_train = y_true[:train_size]
    y_pred_train = y_pred[:train_size]

    geral_dict = {
        'test_metrics': gerenerate_metric_results(y_true_test, y_pred_test),
        'val_metrics': val_result,
        'train_metrics': gerenerate_metric_results(y_true_train, y_pred_train),
        'real_values': y_true,
        'predicted_values': y_pred,
        'params': model_params
    }

    if return_type == 0:
        return geral_dict['test_metrics']
    elif return_type == 1:
        return geral_dict['val_metrics']
    elif return_type == 2:
        return geral_dict['train_metrics']
    elif return_type == 3:


        return save_result(geral_dict, title)


def gerenerate_metric_results(y_true, y_pred):
    return {'MSE': mean_square_error(y_true, y_pred),
            'RMSE':root_mean_square_error(y_true, y_pred),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred),
            'SMAPE':symmetric_mean_absolute_percentage_error(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'theil': u_theil(y_true, y_pred),
            'ARV': average_relative_variance(y_true, y_pred),
            'IA': index_agreement(y_true, y_pred),
            'POCID': prediction_of_change_in_direction(y_true, y_pred)}


def save_result(dict_result, title):

    currentDT = datetime.datetime.now()
    title = 'models_pkl/'+title+"-"+currentDT.strftime('%d%m%y%s')+".pkl"


    with open(title, 'wb') as handle:
        pkl.dump(dict_result, handle)

    print("exported to pkl")
    return title


def open_saved_result(file_name):
    with open(file_name, 'rb') as handle:
        b = pkl.load(handle)
    return b


def fit_sklearn_model(ts, model, test_size, val_size):
    train_size = len(ts) - test_size - val_size
    y_train = ts['actual'][0:train_size]
    x_train = ts.drop(columns=['actual'], axis=1)[0:train_size]

    return model.fit(x_train, y_train)


def predict_sklearn_model(ts, model):

    x = ts.drop(columns=['actual'], axis=1)

    return model.predict(x)


def plot_acf_pacf(ts, name, save_fig):
    lag_acf = acf(ts, nlags=20)
    print(np.mean(lag_acf))
    lag_pacf = pacf(ts, nlags=20, method='ols')
    print(np.mean(lag_pacf))
    # Plot ACF:

    plt.subplot(121)
    N = len(lag_acf)
    values = lag_acf
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, values, width, color='#66cc00')
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96 / np.sqrt(len(ts)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(ts)), linestyle='--', color='gray')
    ax = plt.title('Autocorrelation Function')
    fig = ax.get_figure()

    # Plot PACF:
    plt.subplot(122)
    N = len(lag_pacf)
    values = lag_pacf
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence
    p1 = plt.bar(ind, values, width, color='#66cc00')
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96 / np.sqrt(len(ts)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(ts)), linestyle='--', color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()

    if save_fig:
        fig.savefig(name)



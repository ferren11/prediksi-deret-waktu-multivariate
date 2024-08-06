import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.models import load_model
from keras_tuner import HyperModel
from keras_tuner.tuners import RandomSearch
import os
import plotly.graph_objects as go
import shutil # untuk keperluan reset tuner

tf.random.set_seed(42)

class LSTMHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32),
                       input_shape=(self.input_shape[1], self.input_shape[2])))
        model.add(Dropout(rate=hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)))
        model.add(Dense(self.output_shape))  
        
        model.compile(optimizer=tf.keras.optimizers.Adam(
                          hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                      loss='mean_squared_error')
        return model

def create_lag_features(data, lag=1):
    df_lagged = data.shift(lag)
    df_lagged.columns = [f'{col}_lag{lag}' for col in data.columns]
    return pd.concat([data, df_lagged], axis=1).dropna()
    
def load_data(file_path):
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return df

def preprocess_data(df, df_lagged, train_size):
    train_size = int(train_size * df_lagged.shape[0])
    
    train, test = df_lagged.iloc[:train_size], df_lagged.iloc[train_size:]    
    
    # Extract the dates before scaling
    test_dates = test.index
    
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    
    # X = lagged data; y = real time data
    X_train, y_train = train_scaled[:, len(df.columns):], train_scaled[:, :len(df.columns)]
    X_test, y_test = test_scaled[:, len(df.columns):], test_scaled[:, :len(df.columns)]
    
    X_train = np.array(X_train).reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.array(X_test).reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    return X_train, y_train, X_test, y_test, scaler, test_dates

def hyperparameter_tuning(X_train, y_train, input_shape, output_shape):
#     Clear the tuning directory
    shutil.rmtree('my_dir/lstm_tuning')
    
    hypermodel = LSTMHyperModel()
    hypermodel.input_shape = input_shape
    hypermodel.output_shape = output_shape
    
    tuner = RandomSearch(hypermodel,
                         objective='loss',
                         max_trials=10,
                         executions_per_trial=2,
                         directory='my_dir',
                         project_name='lstm_tuning')
    
    tuner.search(X_train, y_train, epochs=50)
    
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    return best_hps

def train_model(X_train, y_train, units, dropout, learning_rate, epochs):
    model = Sequential()
    model.add(LSTM(units=units, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(y_train.shape[1]))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')
    
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0, shuffle=False)
    
    return model, history

def evaluate_model(model, X_train, y_train, X_test, y_test, scaler, column_names):
    y_train_pred = model.predict(X_train)
    y_train_combined = np.concatenate((y_train, X_train.reshape(X_train.shape[0], X_train.shape[2])), axis=1)
    y_train_pred_combined = np.concatenate((y_train_pred, X_train.reshape(X_train.shape[0], X_train.shape[2])), axis=1)

    y_train_inv = scaler.inverse_transform(y_train_combined)[:, :len(column_names)]
    y_train_pred_inv = scaler.inverse_transform(y_train_pred_combined)[:, :len(column_names)]
    
    train_evaluation_metrics = {}
    for i, var in enumerate(column_names):
        rmse = np.sqrt(mean_squared_error(y_train_inv[:, i], y_train_pred_inv[:, i]))
        mae = mean_absolute_error(y_train_inv[:, i], y_train_pred_inv[:, i])
        mape = mean_absolute_percentage_error(y_train_inv[:, i], y_train_pred_inv[:, i])
        train_evaluation_metrics[var] = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

    y_pred = model.predict(X_test)
    y_test_combined = np.concatenate((y_test, X_test.reshape(X_test.shape[0], X_test.shape[2])), axis=1)
    y_pred_combined = np.concatenate((y_pred, X_test.reshape(X_test.shape[0], X_test.shape[2])), axis=1)

    y_test_inv = scaler.inverse_transform(y_test_combined)[:, :len(column_names)]
    y_pred_inv = scaler.inverse_transform(y_pred_combined)[:, :len(column_names)]

    test_evaluation_metrics = {}
    for i, var in enumerate(column_names):
        rmse = np.sqrt(mean_squared_error(y_test_inv[:, i], y_pred_inv[:, i]))
        mae = mean_absolute_error(y_test_inv[:, i], y_pred_inv[:, i])
        mape = mean_absolute_percentage_error(y_test_inv[:, i], y_pred_inv[:, i])
        test_evaluation_metrics[var] = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

    return train_evaluation_metrics, test_evaluation_metrics, y_test_inv, y_pred_inv

def get_results(history, y_test_inv, y_pred_inv, train_evaluation_metrics, test_evaluation_metrics, column_names, test_dates):
    history_df = pd.DataFrame(history.history)
    df_train_metrics = pd.DataFrame(train_evaluation_metrics).transpose()
    df_eval_metrics = pd.DataFrame(test_evaluation_metrics).transpose()
    df_actual = pd.DataFrame(y_test_inv, columns=[f'Actual {var}' for var in column_names], index=test_dates)
    df_predicted = pd.DataFrame(y_pred_inv, columns=[f'Predicted {var}' for var in column_names], index=test_dates)
    df_comparison = pd.concat([df_actual, df_predicted], axis=1)
    
    return history_df, df_train_metrics, df_eval_metrics, df_actual, df_predicted, df_comparison

def main(file_path, train_size, units, dropout, lr, use_tuning, epochs):
    df = load_data(file_path)
    df_lagged = create_lag_features(df)  # Create lag features
    X_train, y_train, X_test, y_test, scaler, test_dates = preprocess_data(df, df_lagged, train_size)

    input_shape = X_train.shape
    output_shape = y_train.shape[1]

    if use_tuning:
        best_hps = hyperparameter_tuning(X_train, y_train, input_shape, output_shape)
        units = best_hps.get('units')
        dropout = best_hps.get('dropout')
        lr = best_hps.get('learning_rate')
    else:
        units = units
        dropout = dropout
        lr = lr

    model, history = train_model(X_train, y_train, units, dropout, lr, epochs)

    train_evaluation_metrics, test_evaluation_metrics, y_test_inv, y_pred_inv = evaluate_model(
        model, X_train, y_train, X_test, y_test, scaler, df.columns
    )
    
    history_df, df_train_metrics, df_eval_metrics, df_actual, df_predicted, df_comparison = get_results(history, y_test_inv, y_pred_inv, train_evaluation_metrics, test_evaluation_metrics, df.columns, test_dates)
    
    return history_df, df_train_metrics, df_eval_metrics, df_actual, df_predicted, df_comparison, units, dropout, lr

    
if __name__ == "__main__":
    main(file_path="data-all.csv", train_size=0.7, units=100, dropout=0.2, lr=0.001, use_tuning=False, epochs=100)

import multiprocessing
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

np.set_printoptions(threshold=sys.maxsize, suppress=True)
tf.executing_eagerly()


def river_model(shift_period: int = 180, drop: float = 0.5, batch: int = 64, epochs: int = 100):
    # defining the dimensionality of data for use throughout this model
    LENGTH = 8217  # how many columns of data -> len(df.index)
    Y_VARIABLES = 1  # how many variables / labels we want to predict -> len(labels)
    X_VARIABLES = 15  # how many variables we have as features -> len(x_labels)

    df = pd.read_csv('river_df.csv')  # data resulting from processing

    labels = [x for x in df.columns if x.startswith('Hydrometry')]  # sorting for output labels
    x_labels = [x for x in df.columns if x not in labels][1:]  # labels for input
    Y = df.loc[:, labels].shift(shift_period).values.reshape(LENGTH, Y_VARIABLES)  # response variables
    Y = np.nan_to_num(Y)  # filling the unknown variables resulting from shift with 0
    X = df.loc[:, x_labels].values.reshape(LENGTH, 1, X_VARIABLES)  # explanatory variable

    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=617,
                                                        shuffle=True)  # splitting the data

    # building the model
    inputs = keras.Input(
        shape=(1, X_VARIABLES))  # defining input and its shape -> change the second dim depending on # inputs
    x = layers.BatchNormalization(scale=False, center=True)(inputs)  # normalize
    x = layers.Dropout(drop)(x)
    x = layers.Dense(448, 'relu')(x)
    x = layers.Dropout(drop)(x)
    x = layers.Dense(448, 'relu')(x)
    x = layers.Dropout(drop)(x)
    x = layers.Dense(448, 'relu')(x)
    x = layers.Dropout(drop)(x)
    x = layers.LSTM(448, recurrent_dropout=drop)(x)  # implementing lstm for recurrent properties
    x = layers.Dense(448, 'relu')(x)
    x = layers.Dropout(drop)(x)
    x = layers.Dense(448, 'relu')(x)
    x = layers.Dropout(drop)(x)
    x = layers.Dense(448, 'relu')(x)
    x = layers.Dropout(drop)(x)
    outputs = layers.Dense(Y_VARIABLES)(x)  # dim can be changed depending on how many desired output variables
    model = keras.Model(inputs, outputs)
    model.summary()

    # compiling the model
    model.compile(optimizer='rmsprop', loss='mean_squared_error',
                  metrics=[tf.keras.metrics.MeanAbsoluteError(name="mean_absolute_error"),
                           tf.keras.metrics.RootMeanSquaredError(name="RMSE")])

    # best so far: mae=0.4041 - rmse=0.5935 -> for 10 day look-ahead
    #  mae= - rmse= -> 20 day look-ahead

    # train the model
    history = model.fit(x_train, y_train, batch_size=batch, epochs=epochs)

    #model.save('river_model') # saving the model

    loss, mae, rmse = model.evaluate(x_test, y_test)  # returns metrics ab val data
    print()
    print('River model')
    print('Validation:')
    print(f'loss: {loss}')
    print(f'MAE: {mae}')
    print(f'RMSE: {rmse}')

# river_model()



def lake_model(shift_period: int = 180, drop: float = 0.5, batch: int = 32, epochs: int = 100):
    # defining the dimensionality of data for use throughout this model
    LENGTH = 6603  # how many columns of data -> len(df.index)
    Y_VARIABLES = 2  # how many variables / labels we want to predict -> len(labels)
    X_VARIABLES = 6  # how many variables we have as features -> len(x_labels)

    df = pd.read_csv('lake_df.csv')  # data resulting from processing
    # drop the columns that weren't in the dataset's desired output ie DIEC

    labels = [x for x in df.columns if x.startswith('Lake_Level') or x.startswith('Flow_Rate')]  # sorting for output labels
    x_labels = [x for x in df.columns if x not in labels][1:]  # labels for input
    Y = df.loc[:, labels].shift(shift_period).values.reshape(LENGTH, Y_VARIABLES)  # response variables
    Y = np.nan_to_num(Y)  # filling the unknown variables resulting from shift with 0
    X = df.loc[:, x_labels].values.reshape(LENGTH, 1, X_VARIABLES)  # explanatory variable

    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=617,
                                                        shuffle=True)  # splitting the data

    # building the model
    inputs = keras.Input(
        shape=(1, X_VARIABLES))  # defining input and its shape -> change the second dim depending on # inputs
    x = layers.BatchNormalization(scale=False, center=True)(inputs)  # normalize
    x = layers.Dropout(drop)(x)
    x = layers.Dense(448, 'relu')(x)
    x = layers.Dropout(drop)(x)
    x = layers.Dense(448, 'relu')(x)
    x = layers.Dropout(drop)(x)
    x = layers.Dense(448, 'relu')(x)
    x = layers.Dropout(drop)(x)
    x = layers.LSTM(448, recurrent_dropout=drop)(x)  # implementing lstm for recurrent properties
    x = layers.Dense(448, 'relu')(x)
    x = layers.Dropout(drop)(x)
    x = layers.Dense(448, 'relu')(x)
    x = layers.Dropout(drop)(x)
    x = layers.Dense(448, 'relu')(x)
    x = layers.Dropout(drop)(x)
    outputs = layers.Dense(Y_VARIABLES)(
        x)  # dim can be changed depending on how many desired output variables
    model = keras.Model(inputs, outputs)
    model.summary()

    # compiling the model
    model.compile(optimizer='rmsprop', loss='mean_squared_error',
                  metrics=[tf.keras.metrics.MeanAbsoluteError(name="mean_absolute_error"),
                           tf.keras.metrics.RootMeanSquaredError(name="RMSE")])

    # best so far: mae=9.0548 - rmse=15.1545 -> for 10 day look-ahead
    #  mae=5.8189 - rmse=12.7874 -> 20 day look-ahead

    # train the model
    history = model.fit(x_train, y_train, batch_size=batch, epochs=epochs)

    #model.save('lake_model')  # saving the model

    loss, mae, rmse = model.evaluate(x_test, y_test)  # returns metrics ab val data
    print()
    print('Lake model')
    print('Validation:')
    print(f'loss: {loss}')
    print(f'MAE: {mae}')
    print(f'RMSE: {rmse}')


# lake_model()


def waterspring_model(shift_period: int = 180, drop: float = 0.5, batch: int = 32, epochs: int = 100):
    # defining the dimensionality of data for use throughout this model
    LENGTH = 7487 # how many columns of data -> len(df.index)
    Y_VARIABLES = 6 # how many variables / labels we want to predict -> len(labels)
    X_VARIABLES = 14 # how many variables we have as features -> len(x_labels)

    df = pd.read_csv('waterspring_df.csv')  # data resulting from processing

    labels = [x for x in df.columns if x.startswith('Flow_Rate')]  # sorting for output labels
    x_labels = [x for x in df.columns if x not in labels][1:]  # labels for input
    Y = df.loc[:, labels].shift(shift_period).values.reshape(LENGTH, Y_VARIABLES)  # response variables
    Y = np.nan_to_num(Y)  # filling the unknown variables resulting from shift with 0
    X = df.loc[:, x_labels].values.reshape(LENGTH, 1, X_VARIABLES)  # explanatory variable

    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=617,
                                                        shuffle=True)  # splitting the data

    # building the model
    inputs = keras.Input(shape=(1, X_VARIABLES))  # defining input and its shape -> change the second dim depending on # inputs
    x = layers.BatchNormalization(scale=False, center=True)(inputs)  # normalize
    x = layers.Dropout(drop)(x)
    x = layers.Dense(448, 'relu')(x)
    x = layers.Dropout(drop)(x)
    x = layers.Dense(448, 'relu')(x)
    x = layers.Dropout(drop)(x)
    x = layers.Dense(448, 'relu')(x)
    x = layers.Dropout(drop)(x)
    x = layers.LSTM(448, recurrent_dropout=drop)(x)  # implementing lstm for recurrent properties
    x = layers.Dense(448, 'relu')(x)
    x = layers.Dropout(drop)(x)
    x = layers.Dense(448, 'relu')(x)
    x = layers.Dropout(drop)(x)
    x = layers.Dense(448, 'relu')(x)
    x = layers.Dropout(drop)(x)
    outputs = layers.Dense(Y_VARIABLES)(x) # dim can be changed depending on how many desired output variables
    model = keras.Model(inputs, outputs)
    model.summary()

    # compiling the model
    model.compile(optimizer='rmsprop', loss='mean_squared_error',
                  metrics=[tf.keras.metrics.MeanAbsoluteError(name="mean_absolute_error"),
                           tf.keras.metrics.RootMeanSquaredError(name="RMSE")])

    # best so far: mae=5.4235 - rmse=11.7494 -> for 10 day look-ahead
    #  mae=5.8189 - rmse=12.7874 -> 20 day look-ahead

    # train the model
    history = model.fit(x_train, y_train, batch_size=batch, epochs=epochs)

    #model.save('waterspring_model')  # saving the model

    loss, mae, rmse = model.evaluate(x_test, y_test)  # returns metrics ab val data
    print()
    print('Waterspring model')
    print('Validation:')
    print(f'loss: {loss}')
    print(f'MAE: {mae}')
    print(f'RMSE: {rmse}')

# waterspring_model()


def aquifer_model(shift_period: int = 180, drop: float = 0.5, batch: int = 32, epochs: int = 100 ):
    # defining the dimensionality of data for use throughout this model
    LENGTH = 8154  # how many columns of data -> len(df.index)
    Y_VARIABLES = 18  # how many variables / labels we want to predict -> len(labels)
    X_VARIABLES = 55  # how many variables we have as features -> len(x_labels)

    df = pd.read_csv('aquifer_df.csv') # data resulting from processing
    # drop the columns that weren't in the dataset's desired output ie DIEC

    labels = [x for x in df.columns if x.startswith('Depth_to_Groundwater')] # sorting for output labels
    x_labels = [x for x in df.columns if x not in labels][1:] # labels for input
    Y = df.loc[:, labels].shift(shift_period).values.reshape(LENGTH, Y_VARIABLES)  # response variables
    Y = np.nan_to_num(Y)  # filling the unknown variables resulting from shift with 0
    X = df.loc[:, x_labels].values.reshape(LENGTH, 1, X_VARIABLES)  # explanatory variable

    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=617,
                                                        shuffle=True)  # splitting the data

    # building the model
    inputs = keras.Input(shape=(1, X_VARIABLES)) # defining input and its shape -> change the second dim depending on # inputs
    x = layers.BatchNormalization(scale=False, center=True)(inputs) # normalize
    x = layers.Dropout(drop)(x)
    x = layers.Dense(448, 'relu')(x)
    x = layers.Dropout(drop)(x)
    x = layers.Dense(448, 'relu')(x)
    x = layers.Dropout(drop)(x)
    x = layers.Dense(448, 'relu')(x)
    x = layers.Dropout(drop)(x)
    x = layers.LSTM(448, recurrent_dropout=drop)(x) # implementing lstm for recurrent properties
    x = layers.Dense(448, 'relu')(x)
    x = layers.Dropout(drop)(x)
    x = layers.Dense(448, 'relu')(x)
    x = layers.Dropout(drop)(x)
    x = layers.Dense(448, 'relu')(x)
    x = layers.Dropout(drop)(x)
    outputs = layers.Dense(Y_VARIABLES)(x)
    model = keras.Model(inputs, outputs)
    model.summary()

    # compiling the model
    model.compile(optimizer='rmsprop', loss='mean_squared_error',
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mean_absolute_error"),
                 tf.keras.metrics.RootMeanSquaredError(name="RMSE")])

    # best so far: mae=2.9688 - RMSE=4.9911 -> for 10 day look-ahead
    # mae=3.1609 - rmse=5.9243 -> 20 day look-ahead

    # train the model
    history = model.fit(x_train, y_train, batch_size=batch, epochs=epochs)

    #model.save('aquifer_model')  # saving the model

    loss, mae, rmse = model.evaluate(x_test, y_test) # returns metrics ab val data
    print()
    print('Aquifer model')
    print('Validation:')
    print(f'loss: {loss}')
    print(f'MAE: {mae}')
    print(f'RMSE: {rmse}')


#aquifer_model()


# multiprocessing running

if __name__ == '__main__':
    p1 = multiprocessing.Process(target=river_model)
    p2 = multiprocessing.Process(target=lake_model)
    p3 = multiprocessing.Process(target=waterspring_model)
    p4 = multiprocessing.Process(target=aquifer_model)

    p1.start()
    p2.start()
    p3.start()
    p4.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()




def data_processing(
        path: str = '/Users/zacharyhayden/Documents/projects/Python/ML w Python /acea-water-prediction/data',
):
    # this gets the data from all files in the directory
    aquifer_files = [file for file in os.listdir(path) if file.startswith('Aquifer')]
    lake_files = [file for file in os.listdir(path) if file.startswith('Lake')]
    river_files = [file for file in os.listdir(path) if file.startswith('River')]
    waterspring_files = [file for file in os.listdir(path) if file.startswith('Water')]

    # making df's for all the different water bodies
    aquifer_df = pd.DataFrame()
    for file in [Path(path, file) for file in aquifer_files]:
        df = pd.read_csv(file)
        aquifer_df = pd.concat([aquifer_df, df], axis=1)  # combining all the data

    aquifer_df.set_index(['Date'], inplace=True)  # setting index to as date to eliminate the date duplicates
    dex = [dex[0] for dex in aquifer_df.index]  # index values
    dt_dex = pd.to_datetime(dex, format='%d/%m/%Y')  # converting index to datetime format
    aquifer_df.set_index(dt_dex, inplace=True)
    labels = [x for x in df.columns if x.startswith('Depth_to_Groundwater')]

    for col in aquifer_df.columns:  # filtering out the nan vals
        if not col.startswith('Rainfall'):
            aquifer_df = aquifer_df.apply(lambda x: x.fillna(x.mean()), axis=0)

        if col.startswith('Rainfall'):
            aquifer_df[col] = aquifer_df[col].fillna(value=aquifer_df[col].mode())

        if col.startswith('Volume'):
            aquifer_df[col] = np.abs(aquifer_df[col].values)

        if col.startswith('Flow_Rate'):
            aquifer_df[col] = np.abs(aquifer_df[col].values)

        if col.endswith('PAG') or col.endswith('DIEC'): # dropping these cols bc not on output dataset
            aquifer_df.drop(col, axis=1, inplace=True)

    print(aquifer_df.head(20))
    aquifer_df.to_csv('aquifer_df.csv')

    lake_df = pd.DataFrame()
    for file in [Path(path, file) for file in lake_files]:
        df = pd.read_csv(file)
        lake_df = pd.concat([lake_df, df], axis=1)

    lake_df.set_index(['Date'], inplace=True)  # setting index to as date to eliminate the date duplicates
    dex = [dex for dex in lake_df.index]  # index values
    print(dex)
    dt_dex = pd.to_datetime(dex, format='%d/%m/%Y')  # converting index to datetime format
    lake_df.set_index(dt_dex, inplace=True)

    for col in lake_df.columns:  # filtering out the nan vals
        if not col.startswith('Rainfall'):
            lake_df = lake_df.apply(lambda x: x.fillna(x.mean()), axis=0)

        if col.startswith('Rainfall'):
            lake_df[col] = lake_df[col].fillna(value=lake_df[col].mode())

        if col.startswith('Volume'):
            lake_df[col] = np.abs(lake_df[col].values)

        if col.startswith('Flow_Rate'):
            lake_df[col] = np.abs(lake_df[col].values)

    print(lake_df.head(20))
    lake_df.to_csv('lake_df.csv')

    river_df = pd.DataFrame()
    for file in [Path(path, file) for file in river_files]:
        df = pd.read_csv(file)
        river_df = pd.concat([river_df, df], axis=1)

    river_df.set_index(['Date'], inplace=True)  # setting index to as date to eliminate the date duplicates
    dex = [dex for dex in river_df.index]  # index values
    dt_dex = pd.to_datetime(dex, format='%d/%m/%Y')  # converting index to datetime format
    river_df.set_index(dt_dex, inplace=True)

    for col in river_df.columns:  # filtering out the nan vals
        if not col.startswith('Rainfall'):
            river_df = river_df.apply(lambda x: x.fillna(x.mean()), axis=0)

        if col.startswith('Rainfall'):
            river_df[col] = river_df[col].fillna(value=river_df[col].mode())

        if col.startswith('Volume'):
            river_df[col] = np.abs(river_df[col].values)

        if col.startswith('Flow_Rate'):
            river_df[col] = np.abs(river_df[col].values)

    print(river_df.head(20))
    river_df.to_csv('river_df.csv')

    waterspring_df = pd.DataFrame()
    for file in [Path(path, file) for file in waterspring_files]:
        df = pd.read_csv(file)
        waterspring_df = pd.concat([waterspring_df, df], axis=1)

    waterspring_df.set_index(['Date'], inplace=True)  # setting index to as date to eliminate the date duplicates
    dex = [dex[0] for dex in waterspring_df.index]  # index values
    dt_dex = pd.to_datetime(dex, format='%d/%m/%Y')  # converting index to datetime format
    waterspring_df.set_index(dt_dex, inplace=True)

    for col in waterspring_df.columns:  # filtering out the nan vals
        if not col.startswith('Rainfall'):
            waterspring_df = waterspring_df.apply(lambda x: x.fillna(x.mean()), axis=0)

        if col.startswith('Rainfall'):
            waterspring_df[col] = waterspring_df[col].fillna(value=waterspring_df[col].mode())

        if col.startswith('Volume'):
            waterspring_df[col] = np.abs(waterspring_df[col].values)

        if col.startswith('Flow_Rate'):
            waterspring_df[col] = np.abs(waterspring_df[col].values)

    print(waterspring_df.head(20))
    waterspring_df.to_csv('waterspring_df.csv')

# data_processing()

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import BinaryCrossentropy
import sys
import kerastuner

np.set_printoptions(threshold=sys.maxsize, suppress=True)



# defining the dimensionality of data for use throughout this model
LENGTH = 8217  # how many columns of data -> len(df.index)
Y_VARIABLES = 1  # how many variables / labels we want to predict -> len(labels)
X_VARIABLES = 15  # how many variables we have as features -> len(x_labels)

df = pd.read_csv('river_df.csv')  # data resulting from processing

labels = [x for x in df.columns if x.startswith('Hydrometry')]  # sorting for output labels
x_labels = [x for x in df.columns if x not in labels][1:]  # labels for input
Y = df.loc[:, labels].shift(10).values.reshape(LENGTH, Y_VARIABLES)  # response variables
Y = np.nan_to_num(Y)  # filling the unknown variables resulting from shift with 0
X = df.loc[:, x_labels].values.reshape(LENGTH, 1, X_VARIABLES)  # explanatory variable

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=617,
                                                    shuffle=True)  # splitting the data


def river_model(hp):
    # building the model
    inputs = keras.Input(
        shape=(1, X_VARIABLES))  # defining input and its shape -> change the second dim depending on # inputs
    x = layers.BatchNormalization(scale=False, center=True)(inputs)  # normalize
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(units=hp.Int('units', min_value=32, max_value=704, step=32), activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(units=hp.Int('units', min_value=32, max_value=704, step=32), activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(units=hp.Int('units', min_value=32, max_value=704, step=32), activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.LSTM(units=hp.Int('units', min_value=32, max_value=704, step=32), recurrent_dropout=0.5)(x)  # implementing lstm for recurrent properties
    x = layers.Dense(units=hp.Int('units', min_value=32, max_value=704, step=32), activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(units=hp.Int('units', min_value=32, max_value=704, step=32), activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(units=hp.Int('units', min_value=32, max_value=704, step=32), activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(Y_VARIABLES)(x)  # dim can be changed depending on how many desired output variables
    model = keras.Model(inputs, outputs)

    # compiling the model
    model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=[keras.metrics.RootMeanSquaredError(name='RMSE')])

    return model

tuner = kerastuner.tuners.Hyperband(
    river_model,
    objective=kerastuner.Objective('RMSE', direction='min'),
    max_epochs=100,
    executions_per_trial=2,
    directory='model_hypertuning'
)

tuner.search(x_train, y_train)
tuner.results_summary()
models = tuner.get_best_models(num_models=2)
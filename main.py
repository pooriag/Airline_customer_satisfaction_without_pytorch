import random
import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt
import csv_as_dataset

import logistic_regression_model


def data_clean(df):
    df = df.replace(to_replace='satisfied', value=1)
    df = df.replace(to_replace='dissatisfied', value=0)
    ##
    df = df.replace(to_replace='Female', value=0)
    df = df.replace(to_replace='Male', value=1)
    ##
    df = df.replace(to_replace='disloyal Customer', value=0)
    df = df.replace(to_replace='Loyal Customer', value=0)
    ##
    df = df.replace(to_replace='Personal Travel', value=0)
    df = df.replace(to_replace='Business travel', value=1)
    ##
    df = df.replace(to_replace='Eco', value=1)
    df = df.replace(to_replace='Eco Plus', value=2)
    df = df.replace(to_replace='Business', value=0)


    #print(df.isnull().any(axis=1))
    return df

def plot_losses(losses):
    plt.figure()

    for i in range(0, len(losses), 100):
        plt.plot(i, losses[i], 'ro')
    plt.show()

df = pnd.read_csv('Invistico_Airline.csv')
df = data_clean(df)
df['index'] = range(0, len(df))
df.set_index('index', inplace=True)

#cdv.visualize_binary_data_with_respect_to_each_input_features(df, df.columns[len(df.columns) - 1], df.columns[[1]])

cad = csv_as_dataset.csv_dataset("satisfaction")

norm_df = cad.normalize_data(df)
train_data = norm_df[(norm_df.index % 3 == 0) | (norm_df.index % 3 == 1)]
test_data = norm_df[(norm_df.index % 3 == 2) & (norm_df.index % 2 == 0)]
evaluation_data = norm_df[(norm_df.index % 3 == 2) & (norm_df.index % 2 == 1)]

M = logistic_regression_model.logistic_model(len(norm_df.columns) - 1)

cad.train(train_data, 50, 100, M)

test_sample = csv_as_dataset.random_sample(test_data, 20)
for i in range(len(test_sample)):
    print(f'actual value{test_sample["satisfaction"].iloc[i]}')

    x = test_sample["satisfaction"]
    X = np.array(x.values.tolist()[i]).T
    print(f'prediction{M.forward(X)}')
    print("..............................")



plot_losses(M.losses)
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import random
import yfinance as yf
import zipfile
import matplotlib.pyplot as plt
import datetime
import matplotlib.pyplot as plt

from tensorflow import keras
from matplotlib.dates import drange
from pandas import DataFrame
from pandas_datareader import data as pdr
from AlphaNet_Transformer.src.alphanet import AlphaNet, AlphaNetV3, load_model
from AlphaNet_Transformer.src.alphanet.data import TrainValData, TimeSeriesData
from AlphaNet_Transformer.src.alphanet.metrics import UpDownAccuracy

yield_duration = 30
ablation = ""

seed = 100
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

data_all = DataFrame()

path_data_100 = "./data_100_" + str(yield_duration) + "day.csv"
label_name = str(yield_duration) + "-day_return"
if os.path.exists(path_data_100):
    data_all = pd.read_csv(path_data_100)
else:
    names = ['FB', 'OXY', 'AMD', 'AAPL', 'NIO', 'BAC', 'RIG', 'XOM', 'VALE', 'F', 
             'CCL', 'SOFI', 'NVDA', 'SWN', 'T', 'CVX', 'ITUB', 'PLTR', 'NTRA', 'BBD', 
             'MRO', 'GOLD', 'BMBL', 'NOK', 'INTC', 'MSFT', 'WFC', 'UAL', 'C', 'UBER',
             'DIS', 'CLF', 'NCLH', 'DIDI', 'SLB', 'SIRI', 'PBR', 'SQ', 'GRAB', 'HAL',
             'FCEL', 'AUY', 'SNAP', 'DAL', 'ZNGA', 'ET', 'BP', 'AMC', 'DKNG', 'X', 
             'FCX', 'VZ', 'KO', 'KGC', 'PLUG', 'CSX', 'BKR', 'BEKE', 'PFE', 'NLY',
             'TELL', 'KMI', 'BTG', 'GM', 'CMCSA', 'VTRS', 'OPEN', 'MU', 'CCJ', 'MNDT',
             'LYG', 'KOS', 'PYPL', 'CVE', 'CSCO', 'IBN', 'TSLA', 'BABA', 'RBLX', 'LCID',
             'HOOD', 'DVN', 'HPQ', 'ABEV', 'JPM', 'ERIC', 'BTU', 'LUMN', 'TWTR', 'CDEV', 
             'PCG', 'EDU', 'PSXP', 'CS', 'ING', 'NUAN', 'TLRY', 'TME', 'FTCH', 'MARA']

    
    for i in range(len(names)):
        name = names[i]
        yf.pdr_override()
        data = pdr.get_data_yahoo(name, start="2010-01-01", end="2022-04-10")
        data['code']=name
        data['date'] = data.index.strftime('%Y%m%d')
        data[label_name] = (data.Close/data.Close.shift(yield_duration)) -1
        data['return1'] = data.Close.diff()
        data.columns=['open','high','low','close','close-adj','volume','code','date',label_name,'return1'] 
        order = ['code','date',label_name,'close','high','low','open','return1','volume']
        data = data[order]
        data['volume/low'] = data['volume']/data['low']
        data['low/high'] = data['low']/data['high']
        data.drop(data.head(yield_duration).index,inplace=True)

        if i == 0:
            data_all = data
        else:
            data_all = pd.concat([data_all, data])
    
    data_all.to_csv(path_data_100, index=None)
        

df = data_all

# create an empty list
stock_data_list = []

# put each stock into the list using TimeSeriesData() class
security_codes = df["code"].unique()
for code in security_codes:
    table_part = df.loc[df["code"] == code, :]
    stock_data_list.append(TimeSeriesData(dates=table_part["date"].values,                   # date column
                                          data=table_part.iloc[:, 3:].values,                # data columns
                                          labels=table_part[label_name].values)) # label column

date_begin = 20150131
train_length = 800
val_length = 150
test_length = 150
gap_length = yield_duration
train_val_length = train_length + val_length + gap_length

# put stock list into TrainValData() class, specify dataset lengths
train_val_data = TrainValData(time_series_list=stock_data_list,
                              train_length=train_length,   # 1200 trading days for training
                              validate_length=val_length, # 150 trading days for validation
                              history_length=30,   # each input contains 30 days of history
                              sample_step=2,       # jump to days forward for each sampling
                              train_val_gap=gap_length     # leave a 10-day gap between training and validation
                             )

trainval_test_data = TrainValData(time_series_list=stock_data_list,
                              train_length=train_val_length,   # 1200 trading days for training
                              validate_length=test_length, # 150 trading days for validation
                              history_length=30,   # each input contains 30 days of history
                              sample_step=2,       # jump to days forward for each sampling
                              train_val_gap=gap_length     # leave a 10-day gap between training and validation
                             )

model_att_name = './models_weights/' + str(yield_duration) + 'day/model_weights_att_' + str(date_begin) +'_' + str(train_length) + ablation
model_gru_name = './models_weights/' + str(yield_duration) + 'day/model_weights_gru_' + str(date_begin) +'_' + str(train_length)

train, val, dates_info = train_val_data.get(date_begin, order="by_date")
train_val, test, dates_info = trainval_test_data.get(date_begin, order="by_date")

# print(dates_info)

if not os.path.exists(model_att_name + '.index'):
    # get an AlphaNetV3 instance
    model_att = AlphaNet(l2=0.001, dropout=0.1, recurrent_unit='Transformer')

    # you may use UpDownAccuracy() here to evaluate performance
    model_att.compile(metrics=[tf.keras.metrics.RootMeanSquaredError(),
                           UpDownAccuracy()])

    # train
    model_att.fit(train.batch(200).cache(),
              validation_data=val.batch(200).cache(),
              epochs=100)
    
    model_att.save_weights(model_att_name)

    print("\n")
    
print("att test:")
model_att = AlphaNet(l2=0.001, dropout=0.1, recurrent_unit='Transformer')
model_att.compile(metrics=[tf.keras.metrics.RootMeanSquaredError(),
                           UpDownAccuracy()])
model_att.load_weights(model_att_name)
model_att.evaluate(test.batch(200))
print("\n\n\n")

if not os.path.exists(model_gru_name + '.index'):
    # get an AlphaNetV3 instance
    model_gru = AlphaNetV3(l2=0.001, dropout=0, recurrent_unit='GRU')

    # you may use UpDownAccuracy() here to evaluate performance
    model_gru.compile(metrics=[tf.keras.metrics.RootMeanSquaredError(),
                           UpDownAccuracy()])

    # train
    model_gru.fit(train.batch(200).cache(),
              validation_data=val.batch(200).cache(),
              epochs=100)
    
    model_gru.save_weights(model_gru_name)
    
    print("\n")
    
print("gru test:")
model_gru = AlphaNetV3(l2=0.001, dropout=0, recurrent_unit='GRU')
model_gru.compile(metrics=[tf.keras.metrics.RootMeanSquaredError(),
                       UpDownAccuracy()])
model_gru.load_weights(model_gru_name)
model_gru.evaluate(test.batch(200))
print("\n\n\n")



print(date_begin)
print("att test:") 
model_att.load_weights(model_att_name)
model_att.evaluate(test.batch(200))

print("gru test:")
model_gru.load_weights(model_gru_name)
model_gru.evaluate(test.batch(200))
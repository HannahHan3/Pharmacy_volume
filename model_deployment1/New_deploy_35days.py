import os
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime, timedelta
from sklearn import linear_model
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 200)
import warnings
warnings.simplefilter('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor

model_file=pd.read_excel('models.xlsx','Sheet1')


def read_xls():
    df=pd.read_excel('sample_input_35days.xlsx')
    df=df.rename(columns={'Day of Date':'Weekday','Local Sold Total':'Total','Local Sold New':'New','Local Sold New %':'New%','Local Sold Refill':'Refill','Local Sold Refill %':'Refill%','Local Sold CRP Total':'CRP','Local Sold CRP %':'CRP%','RTS Volume':'RTS','RTS %':'RTS%'})
    df.dropna(inplace=True)
    df['Month']=df['Date'].dt.month
    df.set_index('Date',inplace=True)
    return df

input=read_xls()

Weekday=pd.get_dummies(input.Weekday)
input = pd.merge(input,Weekday, right_index=True,left_index=True,how='outer')
Month=pd.get_dummies(input.Month)
input=pd.concat([input,Month],axis=1)
input=input.drop(['Pharmacy'],axis=1)


x = []
temp = []

dowlist = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
for i in dowlist:
    x.append(input[i].tolist()[-1])



(input.index[-1] + timedelta(days=1)).weekday()

temp = [0] * 12
temp[(input.index[-1] + timedelta(days=1)).month-1] = 1
for i in range(len(temp)):
    x.append(temp[i])

temp = []
temp = input['Total'].tolist()
print(temp)
for i in range(len(temp)):
    x.append(temp[::-1][i])

temp = input['New%'].tolist()
for i in range(len(temp)):
    x.append(temp[::-1][i])

# prediction
for d in range(36):
    model=pickle.load(open(model_file['name'].get(key=d),'rb'))
    pred=model.predict(np.reshape(np.asarray(x),(1,-1)))
    print('predicted total drug in Day %d: %d' % (d+35,pred))


import os
import pandas as pd
import pickle
#from sklearn.linear_model import LinearRegression
import numpy as np
#from datetime import datetime, timedelta
#from sklearn import linear_model
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 200)
import warnings
warnings.simplefilter('ignore')
from datetime import datetime, timedelta
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_absolute_error as MAE
#from sklearn.tree import DecisionTreeRegressor
#from sklearn.model_selection import cross_val_score
#from sklearn.ensemble import BaggingRegressor
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.preprocessing import StandardScaler
#from sklearn.linear_model import SGDRegressor
#from sklearn.ensemble import GradientBoostingRegressor
import json
import csv

os.getcwd()
#model_file=pd.read_excel('models.xlsx','Sheet1')

csvFile='Sample_input_35days.csv'
jsonFile ='Sample_input.json'
with open('Sample_input_35days.csv', 'r', encoding = "utf8") as csvfile:
    reader = csv.DictReader(csvfile) 
    rows = list(reader)
csvfile.close()
with open('Sample_input.json', 'w', encoding = "utf8") as jsonfile:
    json.dump(rows,jsonfile)
csvfile.close()




inputs=pd.read_json('sample_input.json',orient='columns')
inputs=inputs.rename(columns={'Day of Date':'Weekday','Local Sold Total':'Total','Local Sold New':'New','Local Sold New %':'New%','Local Sold Refill':'Refill','Local Sold Refill %':'Refill%','Local Sold CRP Total':'CRP','Local Sold CRP %':'CRP%','RTS Volume':'RTS','RTS %':'RTS%'})
inputs.dropna(inplace=True)
inputs['Total']=inputs['Total'].apply(lambda x: x.replace(',',''))
inputs['Total']=pd.to_numeric(inputs['Total'],errors='coerce')
inputs['Month']=inputs['Date'].dt.month
inputs['first_10d']=inputs['Date'].dt.day.apply(lambda x: 1 if x<=10 else 0)
inputs['last_10d']=inputs['Date'].dt.day.apply(lambda x: 1 if x>20 else 0)
inputs['first_10d']=pd.get_dummies(inputs['first_10d'])[1]
inputs['last_10d']=pd.get_dummies(inputs['last_10d'])[1]
inputs.set_index('Date',inplace=True)


Weekday=pd.get_dummies(inputs.Weekday)
inputs = pd.merge(inputs,Weekday, right_index=True,left_index=True,how='outer')
Month=pd.get_dummies(inputs.Month)
inputs=pd.concat([inputs,Month],axis=1)
inputs=inputs.drop(['﻿Pharmacy'],axis=1)


#Day
x = []
temp = []

dowlist = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
for i in dowlist:
    x.append(inputs[i].tolist()[-1])
#Romove Monday
x.pop(0)

#Month
(inputs.index[-1] + timedelta(days=1)).weekday()
temp=[]
temp = [0] * 10
if (inputs.index[-1] + timedelta(days=1)).month-3>=0:
    temp[(inputs.index[-1] + timedelta(days=1)).month-3] = 1
else:
    pass
for i in range(len(temp)):
    x.append(temp[i])
    
#Total:
x.append(inputs['Total'].tolist()[-1])
#first_10_days and last_10_days
x.append(inputs['first_10d'].tolist()[-1])
x.append(inputs['last_10d'].tolist()[-1])

#last 34 days total:
temp = []
temp = inputs['Total'].tolist()
temp.pop(-1)
for i in range(len(temp)):
    x.append(temp[::-1][i])
print(len(x))
# prediction
output={}
model=pickle.load(open('San_Marcos_lasso_df0.sav','rb'))
pred=model.predict(np.reshape(np.asarray(x),(1,-1)))
print('predicted total drug in Day %d: %d' % (1,pred))
output['PRED_d'+str(1)]=list(pred)
json = json.dumps(output)
f = open("output_test.json","w")
f.write(json)
f.close()

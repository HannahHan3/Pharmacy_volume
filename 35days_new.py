#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 22:47:35 2019

@author: huiminhan
"""

import os
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 20)
import warnings
warnings.simplefilter('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
import pickle

path='/Users/andychen/Desktop/UMD/CHIDS/chidsdata'
os.chdir(path)
file = pd.ExcelFile('San Diego MCA Centralized Scheduling Data.xlsx')

def no_weekend(sheet):
    sheet.reset_index()
    return sheet[(sheet.Weekday != 'Saturday') & (sheet.Weekday != 'Sunday')]

def shift_df(df):
    for i in range(34):
        df=pd.concat([df,df['Total'].shift(i+1).rename('Last_'+str(i+1)+"d_Total")],axis=1)
        #df=pd.concat([df,df['New%'].shift(i+1).rename('Last_'+str(i+1)+"d_New%")],axis=1)
    return df

def read_xls(sheet,weekend=True):
    df=pd.read_excel(file,sheet)
    df=df[df.Pharmacy!='Grand Total:']
    df=df.rename(columns={'Day of Date':'Weekday','Local Sold Total':'Total','Local Sold New':'New','Local Sold New %':'New%','Local Sold Refill':'Refill','Local Sold Refill %':'Refill%','Local Sold CRP Total':'CRP','Local Sold CRP %':'CRP%','RTS Volume':'RTS','RTS %':'RTS%'})
    df.dropna(inplace=True)
    df['Month']=df['Date'].dt.month
    df['first_10d']=df['Date'].dt.day.apply(lambda x: 1 if x<=10 else 0)
    df['last_10d']=df['Date'].dt.day.apply(lambda x: 1 if x>20 else 0)
    df['first_10d']=pd.Categorical(df['first_10d'])
    df['last_10d']=pd.Categorical(df['last_10d'])
    df.set_index('Date',inplace=True)
    if weekend==False:
        df=no_weekend(df)
    df=shift_df(df)
    Pharmacy=pd.get_dummies(df.Pharmacy)
    df=pd.concat([Pharmacy,df],axis=1)
    Month=pd.get_dummies(df.Month)
    df=pd.concat([df,Month],axis=1)
    Weekday=pd.get_dummies(df.Weekday)
    df = pd.merge(df,Weekday, right_index=True,left_index=True,how='outer')
    return df

pha_San_Marcos=read_xls('San Marcos 1.1.18 to 4.29.19')
pha_Carlsbad=read_xls('Carlsbad 1.1.18 to 4.29.19',False)
pha_Escondido=read_xls('Escondido 1.1.18 to 4.29.19',False)
pha_Oceanside=read_xls('Oceanside 1.1.18 to 4.29.19',False)
pha_Bernardo_Ctr=read_xls('Bernardo Ctr 1.1.18 to 4.29.',False)
pha_Vista=read_xls('Vista 1.1.18 to 4.29.19',False)


def ylabels(df):
    for i in range(35,71):
        df=pd.concat([df,df['Total'].shift(-i).rename('Next_'+str(i)+"d_Total")],axis=1)
    df.drop(columns=['Weekday','Month','New','Refill','New%','Refill%','CRP','CRP%','RTS','RTS%'],inplace=True)
    return df

pha_San_Marcos_model=ylabels(pha_San_Marcos)
pha_Carlsbad_model=ylabels(pha_Carlsbad)
pha_Escondido_model=ylabels(pha_Escondido)
pha_Oceanside_model=ylabels(pha_Oceanside)
pha_Bernardo_Ctr_model=ylabels(pha_Bernardo_Ctr)
pha_Vista_model=ylabels(pha_Vista)

pha=pd.concat([pha_Bernardo_Ctr_model,pha_Carlsbad_model,pha_Escondido_model,
               pha_Oceanside_model,pha_Vista_model],axis=0)
pha[['Bernardo Center','Carlsbad','Escondido','Oceanside MOB','Vista']]=pha[['Bernardo Center','Carlsbad','Escondido','Oceanside MOB','Vista']].fillna(0)

cols = [
     'Monday','Tuesday','Wednesday','Thursday','Friday',
     'Bernardo Center','Carlsbad','Escondido','Oceanside MOB','Vista','Total',
      1,2,3,4,5,6,7,8,9,10,11,12,
     'first_10d','last_10d',
     'Last_1d_Total','Last_2d_Total','Last_3d_Total','Last_4d_Total','Last_5d_Total','Last_6d_Total','Last_7d_Total',
     'Last_8d_Total','Last_9d_Total','Last_10d_Total','Last_11d_Total','Last_12d_Total','Last_13d_Total','Last_14d_Total',
     'Last_15d_Total','Last_16d_Total','Last_17d_Total','Last_18d_Total','Last_19d_Total','Last_20d_Total','Last_21d_Total',
     'Last_22d_Total','Last_23d_Total','Last_24d_Total','Last_25d_Total','Last_26d_Total','Last_27d_Total','Last_28d_Total',
     'Last_29d_Total','Last_30d_Total','Last_31d_Total','Last_32d_Total','Last_33d_Total','Last_34d_Total',
     'Next_35d_Total','Next_36d_Total','Next_37d_Total','Next_38d_Total','Next_39d_Total','Next_40d_Total','Next_41d_Total',
     'Next_42d_Total','Next_43d_Total','Next_44d_Total','Next_45d_Total','Next_46d_Total','Next_47d_Total','Next_48d_Total',
     'Next_49d_Total','Next_50d_Total','Next_51d_Total','Next_52d_Total','Next_53d_Total','Next_54d_Total','Next_55d_Total',
     'Next_56d_Total','Next_57d_Total','Next_58d_Total','Next_59d_Total','Next_60d_Total','Next_61d_Total','Next_62d_Total',
     'Next_63d_Total','Next_64d_Total','Next_65d_Total','Next_66d_Total','Next_67d_Total','Next_68d_Total','Next_69d_Total',
     'Next_70d_Total']
pha=pha[cols]

pha_San_Marcos_model=pha_San_Marcos_model[[
     'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday', 
      1,2,3,4,5,6,7,8,9,10,11,12,
     'Total','first_10d','last_10d',
     'Last_1d_Total','Last_2d_Total','Last_3d_Total','Last_4d_Total','Last_5d_Total','Last_6d_Total','Last_7d_Total',
     'Last_8d_Total','Last_9d_Total','Last_10d_Total','Last_11d_Total','Last_12d_Total','Last_13d_Total','Last_14d_Total',
     'Last_15d_Total','Last_16d_Total','Last_17d_Total','Last_18d_Total','Last_19d_Total','Last_20d_Total','Last_21d_Total',
     'Last_22d_Total','Last_23d_Total','Last_24d_Total','Last_25d_Total','Last_26d_Total','Last_27d_Total','Last_28d_Total',
     'Last_29d_Total','Last_30d_Total','Last_31d_Total','Last_32d_Total','Last_33d_Total','Last_34d_Total',
     'Next_35d_Total','Next_36d_Total','Next_37d_Total','Next_38d_Total','Next_39d_Total','Next_40d_Total','Next_41d_Total',
     'Next_42d_Total','Next_43d_Total','Next_44d_Total','Next_45d_Total','Next_46d_Total','Next_47d_Total','Next_48d_Total',
     'Next_49d_Total','Next_50d_Total','Next_51d_Total','Next_52d_Total','Next_53d_Total','Next_54d_Total','Next_55d_Total',
     'Next_56d_Total','Next_57d_Total','Next_58d_Total','Next_59d_Total','Next_60d_Total','Next_61d_Total','Next_62d_Total',
     'Next_63d_Total','Next_64d_Total','Next_65d_Total','Next_66d_Total','Next_67d_Total','Next_68d_Total','Next_69d_Total',
     'Next_70d_Total']]

def df_models(df):
    df_dict={}
    for i in range(36):
        df_dict['Model'+str(i+1)]=df.loc[:,:'Last_34d_Total'].join(df['Next_'+str(i+35)+'d_Total']).dropna()
    return df_dict

def split_data(df):
    X_train=df[df.index<'2019'].iloc[:,:-1]
    y_train=df[df.index<'2019'].iloc[:,-1]
    X_valid,X_test,y_valid,y_test=train_test_split(df[df.index>='2019'].iloc[:,:-1],
                                                   df[df.index>='2019'].iloc[:,-1],test_size=0.5,random_state=3)
    return X_train,y_train,X_valid,y_valid,X_test,y_test



path='/Users/andychen/Desktop/UMD/CHIDS/chidsdata/result_new'
os.chdir(path)
# linear regression
def lr_model(df_dict,filename):
    mae_lr={}
    for i in range(36):
        df=df_dict['Model'+str(i+1)]
        lr=LinearRegression(normalize=False)
        lr.fit(split_data(df)[0],split_data(df)[1])
        lr_ypred=lr.predict(split_data(df)[4])
        mae_lr[i]=MAE(split_data(df)[5],lr_ypred)
        #pickle.dump(lr,open(path+'/model_lr/' + filename + '_lr_%s.sav'%('df'+str(i)),'wb'))
    return pd.Series(mae_lr)

result_lr_san_marcos=lr_model(df_models(pha_San_Marcos_model),'San_Marcos')
result_lr_pha=lr_model(df_models(pha),'pha')

result_lr_san_marcos.to_csv('result_lr_san_marcos.csv')
result_lr_pha.to_csv('result_lr_pha.csv')

# lasso regression
def lasso_model(df_dict,filename):
    mae_lasso={}
    alpha_lasso={}
    for i in range(36):
        df=df_dict['Model'+str(i+1)]
        mae_gs={}
        for alpha in np.logspace(-15,7,100):
            lasso=linear_model.Lasso(alpha=alpha)
            lasso.fit(split_data(df)[0],split_data(df)[1])
            lasso_ypred=lasso.predict(split_data(df)[2])
            lasso_mae=MAE(split_data(df)[3],lasso_ypred)
            mae_gs[alpha]=lasso_mae
        alpha_best=min(mae_gs,key=mae_gs.get)
        alpha_lasso[i]=alpha_best
        lasso_best=linear_model.Lasso(alpha=alpha_best)
        lasso_best.fit(split_data(df)[0],split_data(df)[1])
        lasso_ypred=lasso_best.predict(split_data(df)[4])
        mae_lasso[i]=MAE(split_data(df)[5],lasso_ypred)
        print(i)
        pickle.dump(lasso_best,open(path+'/model_lasso/' + filename + '_lasso_%s.sav'%('df'+str(i)),'wb')) 
    return pd.DataFrame({'best_alpha':pd.Series(alpha_lasso),'mae':pd.Series(mae_lasso)})

result_lasso_san_marcos=lasso_model(df_models(pha_San_Marcos_model),'San_Marcos')
result_lasso_pha=lasso_model(df_models(pha),'pha')

result_lasso_san_marcos.to_csv('result_lasso_san_marcos.csv')
result_lasso_pha.to_csv('result_lasso_pha.csv')


# decision tree
def tree_model(df_dict,filename):
    result_dt=pd.DataFrame()
    for i in range(36):
        df=df_dict['Model'+str(i+1)]
        mae_gs=pd.DataFrame()
        for depth in [int(x) for x in np.linspace(start = 1, stop = 20, num = 10)]:
            for leaf in [int(x) for x in np.linspace(start = 1, stop = 10, num = 5)]:
                dt=DecisionTreeRegressor(max_depth=depth,
                                         min_samples_leaf=leaf,
                                         random_state=3)
                dt.fit(split_data(df)[0],split_data(df)[1])
                dt_ypred=dt.predict(split_data(df)[2])
                dt_mae=MAE(split_data(df)[3],dt_ypred)
                mae_gs=mae_gs.append(pd.DataFrame({'max_depth':[depth],
                                                   'min_samples_leaf':[leaf],
                                                   'valid_mae':[dt_mae]}),ignore_index=True)
        parameter_best=mae_gs.loc[mae_gs['valid_mae'].idxmin(),:]
        dt_best=DecisionTreeRegressor(max_depth=int(parameter_best['max_depth']),
                                      min_samples_leaf=int(parameter_best['min_samples_leaf']),
                                      random_state=3)
        dt_best.fit(split_data(df)[0],split_data(df)[1])
        dt_ypred=dt_best.predict(split_data(df)[4])
        mae_dt=pd.Series(MAE(split_data(df)[5],dt_ypred))
        df=parameter_best.append(mae_dt)
        print(i)
        result_dt=result_dt.append(df,ignore_index=True)
        pickle.dump(dt_best,open(path+ '/model_dt/' + filename + '_dt_%s.sav'%('df'+str(i)),'wb')) 
    result_dt.columns=['mae','max_depth','min_samples_leaf','valid_mae']
    return result_dt

result_dt_san_marcos=tree_model(df_models(pha_San_Marcos_model),'San_Marcos')
result_dt_pha=tree_model(df_models(pha),'pha')

result_dt_san_marcos.to_csv('result_dt_san_marcos.csv')
result_dt_pha.to_csv('result_dt_pha.csv')

# bagging
def bagging_model(df_dict,filename):
    result_bagging=pd.DataFrame()
    for i in range(36):
        df=df_dict['Model'+str(i+1)]
        mae_gs=pd.DataFrame()
        for feature in [int(x) for x in np.linspace(start = 1, stop = 10, num = 5)]:
            for sample in [int(x) for x in np.linspace(start = 1, stop = 10, num = 5)]:
                for estimator in [int(x) for x in np.linspace(start = 10, stop = 100, num = 10)]:
                    bagging=BaggingRegressor(max_features=feature,
                                             max_samples=sample,
                                             n_estimators=estimator)
                    bagging.fit(split_data(df)[0],split_data(df)[1])
                    bagging_ypred=bagging.predict(split_data(df)[2])
                    bagging_mae=MAE(split_data(df)[3],bagging_ypred)
                    mae_gs=mae_gs.append(pd.DataFrame({'max_features':[feature],
                                                       'max_samples':[sample],
                                                       'n_estimators':[estimator],
                                                       'valid_mae':[bagging_mae]}),ignore_index=True)
        parameter_best=mae_gs.loc[mae_gs['valid_mae'].idxmin(),:]
        bagging_best=BaggingRegressor(max_features=int(parameter_best['max_features']),
                                      max_samples=int(parameter_best['max_samples']),
                                      n_estimators=int(parameter_best['n_estimators']))
        bagging_best.fit(split_data(df)[0],split_data(df)[1])
        bagging_ypred=bagging_best.predict(split_data(df)[4])
        mae_bagging=pd.Series(MAE(split_data(df)[5],bagging_ypred))
        df=parameter_best.append(mae_bagging)
        print(i)
        result_bagging=result_bagging.append(df,ignore_index=True)
        pickle.dump(bagging_best,open(path+ '/model_bagging/' + filename + '_bagging_%s.sav'%('df'+str(i)),'wb')) 
    result_bagging.columns=['mae','max_features','max_samples','n_estimators','valid_mae']
    return result_bagging

result_bagging_san_marcos=bagging_model(df_models(pha_San_Marcos_model),'San_Marcos')
result_bagging_pha=bagging_model(df_models(pha),'pha')

result_bagging_san_marcos.to_csv('result_bagging_san_marcos.csv')
result_bagging_pha.to_csv('result_bagging_pha.csv')

# random forest
def rf_model(df_dict,filename):
    result_rf=pd.DataFrame()
    for i in range(36):
        df=df_dict['Model'+str(i+1)]
        mae_gs=pd.DataFrame()
        for estimator in [int(x) for x in np.linspace(start = 10, stop = 200, num = 5)]:
            for depth in [int(x) for x in np.linspace(2, 30, num = 5)]:
                for split in [int(x) for x in np.linspace(start = 2, stop = 20, num = 3)]:
                    for leaf in [int(x) for x in np.linspace(start = 1, stop = 20, num = 3)]:
                        rf=RandomForestRegressor(n_estimators=estimator,
                                                 min_samples_split=split,
                                                 min_samples_leaf=leaf,
                                                 max_features='auto',
                                                 max_depth=depth,
                                                 bootstrap=True)
                        rf.fit(split_data(df)[0],split_data(df)[1])
                        rf_ypred=rf.predict(split_data(df)[2])
                        rf_mae=MAE(split_data(df)[3],rf_ypred)
                        mae_gs=mae_gs.append(pd.DataFrame({'n_estimators':[estimator],
                                                           'max_depth':[depth],
                                                           'min_samples_leaf':[leaf],
                                                           'min_samples_split':[split],
                                                           'valid_mae':[rf_mae]}),ignore_index=True)
        parameter_best=mae_gs.loc[mae_gs['valid_mae'].idxmin(),:]
        rf_best=RandomForestRegressor(n_estimators=int(parameter_best['n_estimators']),
                                      min_samples_split=int(parameter_best['min_samples_split']),
                                      min_samples_leaf=int(parameter_best['min_samples_leaf']),
                                      max_features='auto',
                                      max_depth=int(parameter_best['max_depth']),
                                      bootstrap=True)
        rf_best.fit(split_data(df)[0],split_data(df)[1])
        rf_ypred=rf_best.predict(split_data(df)[4])
        mae_rf=pd.Series(MAE(split_data(df)[5],rf_ypred))
        df=parameter_best.append(mae_rf)
        print(i)
        result_rf=result_rf.append(df,ignore_index=True)
        pickle.dump(rf_best,open(path+ '/model_rf/' + filename + '_rf_%s.sav'%('df'+str(i)),'wb')) 
    result_rf.columns=['mae','n_estimators','max_depth','min_samples_leaf','min_samples_split','valid_mae']
    return result_rf

result_rf_san_marcos=rf_model(df_models(pha_San_Marcos_model),'San_Marcos')
result_rf_pha=rf_model(df_models(pha),'pha')

result_rf_san_marcos.to_csv('result_rf_san_marcos.csv')
result_rf_pha.to_csv('result_rf_pha.csv')


# ridge regression
def ridge_model(df_dict,filename):
    mae_ridge={}
    alpha_ridge={}
    for i in range(36):
        df=df_dict['Model'+str(i+1)]
        mae_gs={}
        # [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20, 50, 100, 200, 500]
        for alpha in np.logspace(-15,7,100):
            ridge=linear_model.Ridge(alpha=alpha)
            ridge.fit(split_data(df)[0],split_data(df)[1])
            ridge_ypred=ridge.predict(split_data(df)[2])
            ridge_mae=MAE(split_data(df)[3],ridge_ypred)
            mae_gs[alpha]=ridge_mae
        alpha_best=min(mae_gs,key=mae_gs.get)
        alpha_ridge[i]=alpha_best
        ridge_best=linear_model.Ridge(alpha=alpha_best)
        ridge_best.fit(split_data(df)[0],split_data(df)[1])
        ridge_ypred=ridge_best.predict(split_data(df)[4])
        mae_ridge[i]=MAE(split_data(df)[5],ridge_ypred)
        print(i)
        pickle.dump(ridge_best,open(path+'/model_ridge/' + filename + '_ridge_%s.sav'%('df'+str(i)),'wb')) 
    return pd.DataFrame({'best_alpha':pd.Series(alpha_ridge),'mae':pd.Series(mae_ridge)})

result_ridge_san_marcos=ridge_model(df_models(pha_San_Marcos_model),'San_Marcos')
result_ridge_pha=ridge_model(df_models(pha),'pha')

result_ridge_san_marcos.to_csv('result_ridge_san_marcos.csv')
result_ridge_pha.to_csv('result_ridge_pha.csv')


# stochastic gradient descent
def sgd_model(df_dict,filename):
    result_sgd=pd.DataFrame()
    for i in range(36):
        df=df_dict['Model'+str(i+1)]
        mae_gs=pd.DataFrame()
        for loss in ['squared_loss','huber','epsilon_insensitive']:
            for alpha in [1e-15, 1e-10, 1e-8, 1e-4, 1e-2, 1, 5, 10, 20, 100, 10000]:
                sgd=linear_model.SGDRegressor(loss=loss,
                                              alpha=alpha)
                sgd.fit(split_data(df)[0],split_data(df)[1])
                sgd_ypred=sgd.predict(split_data(df)[2])
                sgd_mae=MAE(split_data(df)[3],sgd_ypred)
                mae_gs=mae_gs.append(pd.DataFrame({'loss':[loss],
                                                   'alpha':[alpha],
                                                   'valid_mae':[sgd_mae]}),ignore_index=True)
        parameter_best=mae_gs.loc[mae_gs['valid_mae'].idxmin(),:]
        sgd_best=linear_model.SGDRegressor(loss=parameter_best['loss'],
                                           alpha=int(parameter_best['alpha']))
        sgd_best.fit(split_data(df)[0],split_data(df)[1])
        sgd_ypred=sgd_best.predict(split_data(df)[4])
        mae_sgd=pd.Series(MAE(split_data(df)[5],sgd_ypred))
        df=parameter_best.append(mae_sgd)
        print(i)
        result_sgd=result_sgd.append(df,ignore_index=True)
        pickle.dump(sgd_best,open(path+ '/model_sgd/' + filename + '_sgd_%s.sav'%('df'+str(i)),'wb')) 
    result_sgd.columns=['mae','loss','alpha','valid_mae']
    return result_sgd

result_sgd_san_marcos=sgd_model(df_models(pha_San_Marcos_model),'San_Marcos')
result_sgd_pha=sgd_model(df_models(pha),'pha')

result_sgd_san_marcos.to_csv('result_sgd_san_marcos.csv')
result_sgd_pha.to_csv('result_sgd_pha.csv')


# gradient boosting
def gb_model(df_dict,filename):
    result_gb=pd.DataFrame()
    for i in range(36):
        df=df_dict['Model'+str(i+1)]
        mae_gs=pd.DataFrame()
        for loss in ['ls','lad','huber']:
            for estimator in [int(x) for x in np.linspace(10, 300, num = 5)]:
                for depth in [int(x) for x in np.linspace(start = 2, stop = 30, num = 4)]:
                    for split in [int(x) for x in np.linspace(start = 2, stop = 20, num = 3)]:
                        gb=GradientBoostingRegressor(loss=loss,
                                                     n_estimators=estimator,
                                                     max_depth=depth,
                                                     min_samples_split=split)
                        gb.fit(split_data(df)[0],split_data(df)[1])
                        gb_ypred=gb.predict(split_data(df)[2])
                        gb_mae=MAE(split_data(df)[3],gb_ypred)
                        mae_gs=mae_gs.append(pd.DataFrame({'loss':[loss],
                                                           'n_estimators':[estimator],
                                                           'max_depth':[depth],
                                                           'min_samples_split':[split],
                                                           'valid_mae':[gb_mae]}),ignore_index=True)
        parameter_best=mae_gs.loc[mae_gs['valid_mae'].idxmin(),:]
        gb_best=GradientBoostingRegressor(loss=parameter_best['loss'],
                                          n_estimators=int(parameter_best['n_estimators']),
                                          max_depth=int(parameter_best['max_depth']),
                                          min_samples_split=int(parameter_best['min_samples_split']))
        gb_best.fit(split_data(df)[0],split_data(df)[1])
        gb_ypred=gb_best.predict(split_data(df)[4])
        mae_gb=pd.Series(MAE(split_data(df)[5],gb_ypred))
        df=parameter_best.append(mae_gb)
        print(i)
        result_gb=result_gb.append(df,ignore_index=True)
        pickle.dump(gb_best,open(path+ '/model_gb/' + filename + '_gb_%s.sav'%('df'+str(i)),'wb')) 
    result_gb.columns=['mae','loss','n_estimators','max_depth','min_samples_split','valid_mae']
    return result_gb

result_gb_san_marcos=gb_model(df_models(pha_San_Marcos_model),'San_Marcos')
result_gb_pha=gb_model(df_models(pha),'pha')

result_gb_san_marcos.to_csv('result_gb_san_marcos.csv')
result_gb_pha.to_csv('result_gb_pha.csv')



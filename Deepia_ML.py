import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import StandardScaler,MinMaxScaler,QuantileTransformer,Normalizer
#allow grow
from xgboost import XGBRegressor
import statsmodels.api as sm
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

df=pd.read_csv('Bittrex_BTCUSD_1h_modified_1.csv')

df.Date=df.Date.apply(lambda x: dt.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
df.Date=pd.to_datetime(df.Date)
df=df.set_index('Date')
df.drop('Symbol',axis=1,inplace=True)
#df['Year']=df.index.year
#df['Month']=df.index.month
#df['Day']=df.index.day
#df['Hour']=df.index.hour

#############
timestep=480
next_time=timestep+1
#######
promedios=[]
distancias=[]
for i in range(len(df['Close'][:])):
    #promedios.append(np.sum(df['Close'][::-1][i:timestep+i])/len(df['Close'][::-1][i:timestep+i]))
    distancias.append(np.std(df['Close'][i:timestep+i]))
    promedios.append(np.mean(df['Close'][i:timestep+i]))

df['promedios']=promedios
#distancia=[abs(df['Close'][::-1][x]-df['promedios'][::-1][x]) for x in range(len(df['Close']))]
df['distancias']=distancias

#----------- train test split

train1,test1=df[['Close','distancias','promedios']][240:],df[['Close','distancias','promedios']][:240]

scaler= StandardScaler()
#scaler = QuantileTransformer()
#scaler=Normalizer()
scaler.fit(train1)

train=scaler.transform(train1[::-1])
test=scaler.transform(test1[::-1])
#np.seed(55)
train=np.array(train1[::-1])
test=np.array(test1[::-1])

real=[]
d2=[]
for i in range(len(train[:,:])):
    try:
        d2.append(train[next_time+i,0])
        real.append(np.array(train[i:timestep+i,:]))
    except:
        print("limite alcanzado {}".format(str(next_time+i)))

#real2=[]
#d3=[]
#for i in range(len(test[:,:1])):
#    try:
#        d3.append(test[next_time+i,0])
#        real2.append(np.array(test[i:timestep+i,:1]))
#    except:
#        print("limite alcanzado {}".format(str(next_time+i)))


r1=XGBRegressor(n_jobs=-1,learning_rate=.5,colsample_bytree=1,max_depth=6,verbosity=2,n_estimators=600)

#r1.fit(np.array(real).reshape(-1,timestep),np.array(d2).reshape(-1,1))
prueba=[]

def probar(dias=30,timestep=timestep,escalar='no'):
    horas=dias*24
    prueba=[]
    prueba.extend(train[-timestep:,0])
    for i in range(horas):
        prueba.append(r1.predict(np.array([prueba[i:]]))[0])
    if escalar=='escalar':
        prueba=scaler.inverse_transform(prueba)
    return prueba

def generar_horarios(prueba,horario=(2020,4,16,22,00,00),horas=len(prueba)):
    horario=dt.datetime(*horario)
    g={'horarios':[],'datos':[]}
    for i in range(horas):
        g['datos'].append(prueba[i])
        g['horarios'].append(horario+dt.timedelta(hours=i))
    return g

#for i in range(17500):
#        prediccion=r1.predict(np.array(prueba[i:]).reshape(1,-1))[0]
#	prueba.append(np.array([prediccion,abs(np.sum([x[0] for x in prueba[i:]])/len(prueba[i:])-prediccion)]))

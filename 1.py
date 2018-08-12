import math
import pandas
import quandl
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing,cross_validation,svm
from sklearn.linear_model import LinearRegression
from matplotlib import style
import datetime
import pickle

style.use('ggplot')


df = quandl.get('WIKI/GOOGL')
#df.Close.plot()
#plot.show()
#print(df.head())
df=df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]

df['HL_PCT']=(df['Adj. High']-df['Adj. Close']) / df['Adj. Close']*100

df['PCT_CHANGE']=(df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100

df=df[['Adj. Close','HL_PCT','PCT_CHANGE','Adj. Volume']]
forecast_col='Adj. Close'

#print(df.head())
df.fillna(-9999,inplace=True)
forecast_out=int(math.ceil(0.1*len(df)))

#print(forecast_out)
df['label']=df[forecast_col].shift(-forecast_out)
#print(df.head())




X=np.array(df.drop(['label'],1))
X=X[:-forecast_out]
X_lately=X[-forecast_out:]
X=preprocessing.scale(X)

df.dropna(inplace=True)
y=np.array(df['label'])
y=np.array(df['label'])

X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)

clf=LinearRegression()
clf.fit(X_train,y_train)
# #with open('E:\\machine\\linearregression.pickle','wb') as f:
#     pickle.dump(df,f)
# pickle_in=open('E:\\machine\\linearregression.pickle.','rb')
clf=LinearRegression(n_jobs=-1)
clf.fit(X_train,y_train)

acuracy=clf.score(X_test,y_test)

#print(acuracy)

forecast_set=clf.predict(X_lately)
print("forecast_set=",forecast_set)
#print(forecast_set)

df['forecast']=np.nan
last_date=df.iloc[-1].name#iloc used for label
last_unix=last_date.timestamp()
one_day=86400
next_unix=last_unix+one_day

for i in forecast_set:
    next_date=datetime.datetime.fromtimestamp(next_unix)
    next_unix+=one_day
    df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)]+[i] #loc is used for column


df['Adj. Close'].plot()
df['forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()





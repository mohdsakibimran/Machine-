import pylab
from matplotlib.dates import date2num, WeekdayLocator, DateFormatter
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
from matplotlib.ticker import FuncFormatter, MaxNLocator
import matplotlib.pyplot as plt
from nsepy import get_history
import pandas as pd
from datetime import date
import numpy as np
sym=["TCS","INFY",]
for i in sym:
    result = pd.DataFrame(get_history(symbol=i, start=date(2015,1,1), end=date(2015,12,30)))
    result_close=pd.DataFrame(result.Close)

    def moving_avg(data,window):
        weights=np.repeat(1.0,window)/window
        smas=np.convolve(data,weights,'valid')
        return smas

    # result_close['7*4']  = moving_avg(result.Close,window = 7*4)
    # result_close['16*7'] = moving_avg(result.Close,window=7 * 16)
    # result_close['28*7'] = moving_avg(result.Close,window=7 * 28)
    # result_close['7*40'] = moving_avg(result.Close, window=7 * 40)
    # result_close['7*52'] = moving_avg(result.Close, window=7 * 52)
    # print(result_close['7*4'])


    #we can do samething effectiveway using rolling mean

    #handle data due to holiday, I am using previous day data for equality
    result.asfreq('D', method='pad')
    result_close=pd.DataFrame(result.Close)
    result_close['7*4']  = result_close.Close.rolling(7*4).mean()
    result_close['7*16'] = result_close.Close.rolling(7 * 16).mean()
    result_close['7*28'] = result_close.Close.rolling(7 * 28).mean()
    result_close['7*40'] = result_close.Close.rolling(7 * 40).mean()
    result_close['7*52'] = result_close.Close.rolling(7 * 52).mean()

    # # handle data due to holiday, I am using previous date data for equality
    # result.asfreq('D',method='pad')




    result['pct_volume_change']=pd.DataFrame(result.Volume)
    #log compute log chnage or change accorrding to previous days
    #add a column for volume percent change
    #result['pct_volume_change']=(((result['Volume']-result['Volume']).shift()/result['change'])*100)

    # add dummy for volume shock
    result['dummy_volume_change_shock']=(((result['Volume']-result['Volume'].shift())/result['Volume'])<=0.1).astype(int)
    #print(result.head())

    #Closing price shock
    # result['pct_close_change']=np.log(((result['Close']-result['Close'].shift())/result['Close'])*100)
    result['dummy_close_change_shock']=(((result['Close']-result['Close'].shift())/result['Close'])<=.02).astype(int)

    #Closing price shock without Volume Shock
    result['closing_wout_volume']=((((result['Close']-result['Close'].shift())/result['Close'])<=.02)&(((result['Volume']-result['Volume'].shift())/result['Volume'])<=0.1)).astype(int)
    print(result.head())



    # visualization of data


    fig, ax=plt.subplots(figsize=(50,40))
    ax.plot(result_close['7*4'],  label= "simplemoving4week",color='c')
    ax.plot(result_close['7*16'], label= "simplemoving16week",color='y')
    ax.plot(result_close['7*28'], label="simplemoving28week",color='m')
    ax.plot(result_close['7*40'], label="simplymoving28week",color='k')
    ax.plot(result_close['7*52'], label="simplymoving28week", color='k')



    # ax.legend(loc='mva')
    # plt.xlabel('timeseries')
    # plt.ylabel('Closing price')
    # plt.title('National Stock Exchange')
    # fig,ax2=plt.subplots(figsize=(16,9))
    # ax2.plot(result['dummy_volume_change_shock'],label="dummy_volume_change")
    plt.legend(loc="change")
    plt.show()


 



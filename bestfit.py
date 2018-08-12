import matplotlib.pyplot as plt
import math
from statistics import mean
import numpy as np
from matplotlib import style
import random
style.use("fivethirtyeight")
#xs=np.array([1,2,3,4,5,6],dtype=np.float64)
#ys=np.array([5,4,6,5,6,7],dtype=np.float64)
def best_fit_slope_intercept(xs,ys):
    m=((mean(xs)*mean(ys)-mean(xs*ys))/((mean(xs)*mean(xs))-mean(xs*xs)))
    b=mean(ys)-m*mean(xs)
    return m,b

def squared_error(ys_orig,ys_line):
    return sum((ys_line-ys_orig)**2)

def create_dataset(hm,variance,step=2,correlation=False):
    val=1
    ys=[]
    for i in range(hm):
        y=val+random.randrange(-variance,variance)
        ys.append(y)
        if correlation and correlation=='pos':
            val+=step
        elif correlation and correlation=='neg':
            val-=step
    xs=[i for i in range(len(ys))]

    return np.array(xs,dtype=np.float64),np.array(ys,dtype=np.float64)

def determenation_sqr_error(ys_orig,ys_line):
    y_mean_line=[mean(ys_orig) for y in (ys_orig)]
    sqr_er_re=squared_error(ys_orig,ys_line)
    sqr_er_ymean=squared_error(y_mean_line,ys_line)
    return 1-(sqr_er_re/sqr_er_ymean)

xs,ys=create_dataset(40,40,2,correlation=False)
m,b=best_fit_slope_intercept(xs,ys)


regressionline=np.array([(m*x+b) for x in (xs)])
r_squared_error=determenation_sqr_error(ys,regressionline)
print(r_squared_error)
predict_x=8
predict_y=(m*predict_x)+b

plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y,color='g')
plt.show()
plt.plot(xs,regressionline)
plt.show()
from sklearn.cluster import KMeans
from sklearn import preprocessing
import numpy as np
import pandas as pd
df=pd.read_excel('titanic3.xls')
df.drop(['body','name'],1,inplace=True)
#df.convert_objects(convert_numeric=True)
df.fillna(0,inplace=True)
def handle_nonnumerical_data(df):
    columns=df.columns.values
    for column in columns:
        text_digit_vals={}
        def convert_to_int(val):
            return text_digit_vals[val]
        if (df[column].dtype!=np.int64 and df[column].dtype!=np.float64):
            x=0
            column_content=df[column].values.tolist()
            unique_elements=set(column_content)
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique]=x
                    x+=1
            df[column]=list(map(convert_to_int,df[column]))
    return df
df.drop(['ticket'],1,inplace=True)
df=handle_nonnumerical_data(df)
X=np.array(df.drop(['survived'],1).astype(float))
y=np.array(df['survived'])
X=preprocessing.scale(X)
clf=KMeans(n_clusters=2)
clf.fit(X)
correct=0
for i in range(len(X)):
    predict_me=np.array(X[i].astype(float))
    print(predict_me)
    predict_me=predict_me.reshape(-1,len(predict_me))
    print(predict_me,end=",")
    pre=clf.predict(predict_me)
    if(pre[0]==y[1]):
        correct+=1
print(correct/len(X))



import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.mlab import stineman_interp
import pandas as pd
import requests
import zipfile as zi 
import StringIO as sio
from sklearn import datasets, linear_model
import scipy.special as scsp
import statsmodels.api as sm
import math
import seaborn as sns


url = 'http://realized.oxford-man.ox.ac.uk/media/1366/'
url += 'oxfordmanrealizedvolatilityindices.zip'
data = requests.get(url, stream=True).content
z = zi.ZipFile(sio.StringIO(data))
z.extractall()

df = pd.read_csv('OxfordManRealizedVolatilityIndices.csv', index_col=0, header=2 )
rv1 = pd.DataFrame(index=df.index)
for col in df.columns:
    if col[-3:] == '.rk':
        rv1[col] = df[col]
rv1.index = [dt.datetime.strptime(str(date), "%Y%m%d") for date in rv1.index.values]

spx = pd.DataFrame(rv1['SPX2.rk'])
spx.plot(color='red', grid=True, title='SPX realized variance', figsize=(16, 9), ylim=(0,0.003))


# replace with YFinance
# SPX = web.DataReader(name = '^GSPC',data_source = 'yahoo', start='2000-01-01')
# SPX = SPX['Adj Close']
# SPX.plot(title='SPX',figsize=(14, 8));


spx['sqrt']= np.sqrt(spx['SPX2.rk'])
spx['log_sqrt'] = np.log(spx['sqrt'])

def del_raw(q, x): 
    return [np.mean(np.abs(spx['log_sqrt'] - spx['log_sqrt'].shift(lag))**q) for lag in x]

plt.figure(figsize=(8, 8))
plt.xlabel('$log(\Delta)$')
plt.ylabel('$log\  m(q.\Delta)$')
plt.ylim=(-3, -.5)

zeta_q = list()
q_vec = np.array([.5, 1, 1.5, 2, 3])
x = np.arange(1, 100)

for q in q_vec:
    plt.plot(np.log(x), np.log(del_raw(q, x)), 'o') 
    model = np.polyfit(np.log(x), np.log(del_raw(q, x)), 1)
    plt.plot(np.log(x), np.log(x) * model[0] + model[1])
    zeta_q.append(model[0])
    
print(zeta_q)


plt.figure(figsize=(8,8))
plt.xlabel('q')
plt.ylabel('$\zeta_{q}$')
plt.plot(q_vec, zeta_q, 'or')

line = np.polyfit(q_vec[:4], zeta_q[:4],1)
plt.plot(q_vec, line[0] * q_vec + line[1])
h_est= line[0]
print(h_est)
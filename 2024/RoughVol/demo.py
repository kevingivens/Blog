import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.mlab import stineman_interp
import pandas as pd
from sklearn import datasets, linear_model
import scipy.special as scsp
import statsmodels.api as sm
import seaborn as sns


# Taken from https://tpq.io/p/rough_volatility_with_python.html  Jim Gatheral
"""
AEX: AEX index
AORD: All Ordinaries
BFX: Bell 20 Index
BSESN: S&P BSE Sensex
BVLG: PSI All-Share Index (excluded because this index is observed from 2012-10-15)
BVSP: BVSP BOVESPA Index
DJI: Dow Jones Industrial Average
FCHI: CAC 40
FTMIB: FTSE MIB
FTSE: FTSE 100
GDAXI: DAX
GSPTSE: S&P/TSX Composite index
HSI: HANG SENG Index
IBEX: IBEX 35 Index
IXIC: Nasdaq 100
KS11: Korea Composite Stock Price Index (KOSPI)
KSE: Karachi SE 100 Index
MXX: IPC Mexico
N225: Nikkei 225
NSEI: NIFTY 50
OMXC20:
OMX Copenhagen 20 Index
OMXHPI: OMX Helsinki All Share Index
OMXSPI: OMX Stockholm All Share Index
OSEAX: Oslo Exchange All-share Index
RUT: Russel 2000
SMSI: Madrid General Index
SPX: S&P 500 Index
SSEC: Shanghai Composite Index
SSMI: Swiss Stock Market Index
STI: Straits Times Index (excluded because this index is NA in the period)
STOXX50E: EURO STOXX 50
"""


# Replace this
# url = 'http://realized.oxford-man.ox.ac.uk/media/1366/'
# url += 'oxfordmanrealizedvolatilityindices.zip'
# data = requests.get(url, stream=True).content
# z = zi.ZipFile(sio.StringIO(data))
# z.extractall()

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



def dlsig2(sig, x, pr=False):
    if pr:
        a= np.array([(sig-sig.shift(lag)).dropna() for lag in x])
        a=a ** 2
        print(a.info())
    return [np.mean((sig-sig.shift(lag)).dropna() ** 2) for lag in x]

h = []
nu = []

for col in rv1.columns:
    sig = rv1[col]
    sig = np.log(np.sqrt(sig))
    sig = sig.dropna()
    model = np.polyfit(np.log(x), np.log(dlsig2(sig, x)), 1)
    nu.append(np.sqrt(np.exp(model[1])))
    h.append(model[0]/2.)
    
OxfordH = pd.DataFrame({'names':rv1.columns, 'h_est': h, 'nu_est': nu})

def plot_scaling(j, scaleFactor):
    col_name = rv1.columns[j]
    v = rv1[col_name]
    x = np.arange(1,101)
    
    def x_del(x, lag):
        return x-x.shift(lag)
    
    def sdl(lag):
        return (x_del(np.log(v), lag)).std()
    
    sd1 = (x_del(np.log(v), 1)).std()
    h = OxfordH['h_est'][j]
    f, ax = plt.subplots(2,2,sharex=False, sharey=False, figsize=(10, 10))
    
    for i_0 in range(0, 2):
        for i_1 in range(0, 2):
            la = scaleFactor ** (i_1*1+i_0*2)
        
            hist_val = x_del(np.log(v), la).dropna()
            std = hist_val.std()
            mean = hist_val.mean()
        
            ax[i_0][i_1].set_title('Lag = %s Days' %la)
            n, bins, patches = ax[i_0][i_1].hist(hist_val.values, bins=100,
                                   normed=1, facecolor='green',alpha=0.2)
            ax[i_0][i_1].plot(bins, mlab.normpdf(bins,mean,std), "r")
            ax[i_0][i_1].plot(bins, mlab.normpdf(bins,0,sd1 * la ** h), "b--")
            hist_val.plot(kind='density', ax=ax[i_0][i_1])
        
def c_tilde(h):
    return scsp.gamma(3./2. - h)/scsp.gamma(h + 1./2.) * scsp.gamma(2.- 2.*h)

def forecast_XTS(rvdata, h, date, nLags, delta, nu):
    i = np.arange(nLags)
    cf = 1./((i + 1. / 2.) ** (h + 1. / 2.) * (i + 1. / 2. + delta))
    ldata = rvdata.truncate(after=date)
    l = len(ldata)
    ldata = np.log(ldata.iloc[l - nLags:])
    ldata['cf'] = np.fliplr([cf])[0]
    # print ldata
    ldata = ldata.dropna()
    fcst = (ldata.iloc[:, 0] * ldata['cf']).sum() / sum(ldata['cf'])
    return math.exp(fcst + 2 * nu ** 2 * c_tilde(h) * delta ** (2 * h))


rvdata = pd.DataFrame(rv1['SPX2.rk'])
nu  = OxfordH['nu_est'][0] # Vol of vol estimate for SPX
h = OxfordH['h_est'][0] 
n = len(rvdata)
delta = 1
nLags = 500
dates = rvdata.iloc[nLags:n-delta].index
rv_predict = [forecast_XTS(rvdata, h=h, date=d, nLags=nLags,
                           delta=delta, nu=nu) for d in dates]
rv_actual = rvdata.iloc[nLags+delta:n].values


plt.figure(figsize=(8, 8))
plt.plot(rv_predict, rv_actual, 'bo')

plt.figure(figsize=(11, 6))
vol_actual = np.sqrt(np.multiply(rv_actual,252))
vol_predict = np.sqrt(np.multiply(rv_predict,252))
plt.plot(vol_actual, "b")
plt.plot(vol_predict, "r")

def xi(date, tt, nu,h, tscale):  # dt=(u-t) is in units of years
    rvdata = pd.DataFrame(rv1['SPX2.rk'])
    return [ forecast_XTS(rvdata,h=h,date=date,nLags=500,delta=dt*tscale,nu=nu) for dt in tt]

nu = OxfordH["nu_est"][0]
h = OxfordH["h_est"][0]

def varSwapCurve(date, bigT, nSteps, nu, h, tscale, onFactor):
  # Make vector of fwd variances
  tt = [ float(i) * (bigT) / nSteps for i in range(nSteps+1)]
  delta_t = tt[1]
  xicurve = xi(date, tt, nu, h, tscale)
  xicurve_mid = (np.array(xicurve[0:nSteps]) + np.array(xicurve[1:nSteps+1])) / 2
  xicurve_int = np.cumsum(xicurve_mid) * delta_t
  varcurve1 = np.divide(xicurve_int, np.array(tt[1:]))
  varcurve = np.array([xicurve[0],]+list(varcurve1))
  varcurve = varcurve * onFactor * tscale #  onFactor is to compensate for overnight moves
  res = pd.DataFrame({"texp":np.array(tt), "vsQuote":np.sqrt(varcurve)})
  return res

def varSwapForecast(date,tau,nu,h,tscale,onFactor):
  vsc = varSwapCurve(date, bigT=2.5, nSteps=100, nu=nu, h=h,
                    tscale=tscale, onFactor=onFactor) # Creates the whole curve
  x = vsc['texp']
  y = vsc['vsQuote']
  res = stineman_interp(tau,x,y,None)

  return(res)

# Test the function

tau = (.25,.5,1,2)
date = dt.datetime(2008,9,8)
varSwapForecast(date,tau,nu=nu,h=h,tscale=252,onFactor=1)

def xip(date, tt, nu,h, tscale):  # dt=(u-t) is in units of years
    rvdata = pd.DataFrame(rv1['SPX2.rk'])
    rvdata_p = rvdata.drop((dt.datetime(2010, 5, 6)), axis=0)
    return [ forecast_XTS(rvdata_p, h=h, date=date,nLags=500,
                          delta=delta_t * tscale, nu=nu) for delta_t in tt]

nu = OxfordH["nu_est"][0]
h = OxfordH["h_est"][0]

def varSwapCurve_p(date, bigT, nSteps, nu, h, tscale, onFactor):
  # Make vector of fwd variances
  tt = [ float(i) * (bigT) / nSteps for i in range(nSteps+1)]
  delta_t = tt[1]
  xicurve = xip(date, tt, nu, h, tscale)
  xicurve_mid = (np.array(xicurve[0:nSteps]) + np.array(xicurve[1:nSteps + 1])) / 2
  xicurve_int = np.cumsum(xicurve_mid) * delta_t
  varcurve1 = np.divide(xicurve_int, np.array(tt[1:]))
  varcurve = np.array([xicurve[0],]+list(varcurve1))
  varcurve = varcurve * onFactor * tscale #  onFactor is to compensate for overnight moves
  res = pd.DataFrame({"texp":np.array(tt), "vsQuote":np.sqrt(varcurve)})
  return(res)

def varSwapForecast_p(date, tau, nu, h, tscale, onFactor):
  vsc = varSwapCurve_p(date, bigT=2.5, nSteps=100, nu=nu, h=h,
                    tscale=tscale, onFactor=onFactor) # Creates the whole curve
  x = vsc['texp']
  y = vsc['vsQuote']
  res = stineman_interp(tau, x, y, None)

  return(res)

# Test the function

tau = (.25, .5 ,1, 2)
date = dt.datetime(2010, 5, 10)
varSwapForecast_p(date, tau, nu=nu, h=h, tscale=252, onFactor=1. / (1 - .35))
import numpy as np
import pandas as pd
from matplotlib.pyplot import *
import statsmodels.api as sma
import statsmodels.stats as sms


import warnings
warnings.filterwarnings('ignore')

# Load the data, and shift the data for future use.
df = pd.read_excel('Case_GDP-round2.xls', index_col=0)

# Add Lags
fullDf = pd.concat([df, df.shift(1), df.shift(2)], axis=1)
cols = []
for lg in range(0,3):
    for i in df.columns:
        if lg == 0:
            cols.append(i)
        else:
            cols.append(i+'(-'+str(lg)+')')
fullDf.columns = cols
print(fullDf.head(3))

print('(a):')
print('Logliklihood Ratio Test')

(start, end) = ('1951Q1', '2010Q4')

x = fullDf[['li1(-1)', 'li2(-1)']]
xA = x[(x.index >= start) & (x.index <= end)]

y = fullDf[['GDPIMPR']]
yA = y[(y.index >= start) & (y.index <= end)]

fitA = sma.Logit(endog=yA, exog=sma.add_constant(xA)).fit()
fitA_C1 = sma.Logit(endog=yA, exog=sma.add_constant(xA[['li1(-1)']])).fit()
fitA_C2 = sma.Logit(endog=yA, exog=sma.add_constant(xA[['li2(-1)']])).fit()

print(fitA.summary2())

print('Loglikelihood Values: [%.3f, %.3f, %.3f, %.3f]' % (fitA.llnull, fitA_C1.llf, fitA_C2.llf, fitA.llf))

print('Likelihood Ratio P-Values:\n [Const+li1=%.6f, Const+li2=%.6f, Const+li1+li2=%.6f]'
      % (fitA_C1.llr_pvalue, fitA_C2.llr_pvalue, fitA.llr_pvalue))

print('(b):')
print('McFaddens R^2 = 1 - LL_Model/LL_Intercept.')

x = fullDf[['li1(-1)', 'li2(-1)', 'li1(-2)', 'li2(-2)']]
xB = x[(x.index >= start) & (x.index <= end)]

y = fullDf[['GDPIMPR']]
yB = y[(y.index >= start) & (y.index <= end)]

fitB = sma.Logit(endog=yB, exog=sma.add_constant(xB)).fit()
fitB_C11 =  sma.Logit(endog=yB, exog=sma.add_constant(xB[['li1(-1)','li2(-1)']])).fit()
fitB_C12 =  sma.Logit(endog=yB, exog=sma.add_constant(xB[['li1(-1)','li2(-2)']])).fit()
fitB_C21 =  sma.Logit(endog=yB, exog=sma.add_constant(xB[['li1(-2)','li2(-1)']])).fit()
fitB_C22 =  sma.Logit(endog=yB, exog=sma.add_constant(xB[['li1(-2)','li2(-2)']])).fit()

LL_null = fitB.llnull  # -152.763
LL_c11  = fitB_C11.llf # -134.178
LL_c12  = fitB_C12.llf # -134.126
LL_c21  = fitB_C21.llf # -130.346
LL_c22  = fitB_C22.llf # -130.461

print('Log Likelihoods:')
print('LL_null=%.4f, LL_c11=%.4f, LL_c12=%.4f, LL_c21=%.4f, LL_c22=%.4f' % (LL_null, LL_c11, LL_c12, LL_c21, LL_c22))

MR2 = [1.0-LL_c11/LL_null, 1.0-LL_c12/LL_null, 1.0-LL_c21/LL_null, 1.0-LL_c22/LL_null]

print('McFaddens R Squared:')
print(', '.join(['%.4f' % e for e in MR2]))

print('(c):')
print('Prediction-realization table and hit rate, using a cut-off value of 0.5.')

print(fitB_C21.summary2())

x = fullDf[['li1(-2)','li2(-1)']]
xC = x[x.index > end]

predC = fitB_C21.predict(sma.add_constant(xC))
predCTable = fitB_C21.pred_table(threshold=0.5)

print('Prediction Realization Table:', predCTable)
print('Hit Rate :')

print('(d):')
print('ADF Test for Log GDP.')

x = fullDf[['LOGGDP']]
xD = x[(x.index >= start) & (x.index <= end)]

loggdp_ADF = sma.tsa.stattools.adfuller(xD['LOGGDP'], maxlag=1, autolag=None, regression='ct', regresults=True)

print('Statistic:', loggdp_ADF[0], ',  P-value:', loggdp_ADF[1])
print('Confidence Levels:', loggdp_ADF[2])

print(loggdp_ADF[3].resols.summary2())
# [x1, x2, const, x3] = ['LOG GDP lag1', 'Diff LOG GDP lag1', 'Constant', 'Trend']
# print(loggdp_ADF[3].resols.summary2(xname=['LOG GDP lag1','Diff LOG GDP lag1','Constant','Trend']))


print('(e):')
print('Growthrate(t) ~ a + rho*GrowthRate(t-1) + beta1*li1(t-k1) + beta2*li2(t-k2)')

# k1,k2 in {1,2}
# Presents R^2 of 4 models and the coeffs of the model with the largest R^2.

x1 = fullDf[['GrowthRate(-1)', 'li1(-1)','li2(-1)']]
x11 = x1[(x1.index >= start) & (x1.index <= end)]

x2 = fullDf[['GrowthRate(-1)', 'li1(-2)', 'li2(-1)']]
x12 = x2[(x2.index >= start) & (x2.index <= end)]

x3 = fullDf[['GrowthRate(-1)', 'li1(-1)', 'li2(-2)']]
x21 = x3[(x3.index >= start) & (x3.index <= end)]

x4 = fullDf[['GrowthRate(-1)', 'li1(-2)', 'li2(-2)']]
x22 = x4[(x4.index >= start) & (x4.index <= end)]

y1 = fullDf[['GrowthRate']]
y11 = y1[(y1.index >= start) & (y1.index <= end)]
y12 = y11.copy()
y21 = y11.copy()
y22 = y11.copy()

modE11 = sma.OLS(endog=y11, exog=sma.add_constant(x11)).fit()
modE12 = sma.OLS(endog=y12, exog=sma.add_constant(x12)).fit()
modE21 = sma.OLS(endog=y21, exog=sma.add_constant(x21)).fit()
modE22 = sma.OLS(endog=y22, exog=sma.add_constant(x22)).fit()

print('R Squares: [mod11=%.6f, mod12=%.6f, mod21=%.6f, mod22=%.6f]'
      % (modE11.rsquared, modE12.rsquared, modE21.rsquared, modE22.rsquared))

print(modE11.params)

print('(f):')
print('Breusch Godfrey Test for model (k1=k2=1) on serial autocorrelation')

bgall = sms.diagnostic.acorr_breush_godfrey(modE11, nlags=1, store=True)
print('lm(Lagrange multiplier test statistic)=%.6f' % bgall[0])
print('lmpval(p-value for Lagrange multiplier test)=%.6f' % bgall[1])
print('fval(fstatistic for F test)=%.6f' % bgall[2])
print('fpval(pvalue for F test)=%.6f' % bgall[3])

print('(g):')
print('Forecast for the Growth Rate.')

x = fullDf[['GrowthRate(-1)', 'li1(-1)', 'li2(-1)']]
xG = x[x.index > end]
xGR = pd.DataFrame(fullDf[fullDf.index > end]['GrowthRate'])

predG = modE11.predict(sma.add_constant(xG))

predGdf = pd.DataFrame(predG, index=xG.index)
outDf = pd.concat([predGdf, xGR], axis=1, join='outer')
outDf.columns = ['Predicted', 'Actual']

fig, ax = subplots(1, 1, figsize=(12,6))
outDf.plot(ax=ax, title='Growth Rate Estimates')
show()

# mae = np.sum(np.absolute(predGdf.values-xGR.values))/len(predGdf.values)
rmse = np.sqrt(((predGdf.values-xGR.values) ** 2).mean())
print("RMSE=%.6f" % rmse)

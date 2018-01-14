import datetime
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

df = pd.read_excel('TestExer6-CARS-round2.xlsx')

df['time'] = df['YYYY-MM'].apply(datetime.datetime.strptime, args=('%YM%m',))
df['y'] = df['TOYOTA_SA']
df['x'] = df['OTHER_SA']
df['share'] = df['y']/(df['y']+df['x'])

df['y-1'] = df['y'].shift(1)
df['dy'] = df['y'].diff()
df['dy-1'] = df['dy'].shift(1)
df['dy-2'] = df['dy'].shift(2)
df['dy-3'] = df['dy'].shift(3)
df['dy-4'] = df['dy'].shift(4)
df['dy-5'] = df['dy'].shift(5)
df['dy-10'] = df['dy'].shift(10)
df['dy-12'] = df['dy'].shift(12)

df['x-1'] = df['x'].shift(1)
df['dx'] = df['x'].diff()
df['dx-1'] = df['dx'].shift(1)
df['dx-2'] = df['dx'].shift(2)
df['dx-3'] = df['dx'].shift(3)

df['ecm1'] = df['y-1'] - 0.45*df['x-1']

n = len(df)
n2 = 12
df1 = df.head(n-n2)
df2 = df.tail(n2) # df.iloc[-12:]


print("\n(a):")
print(df.head())

fig,ax = plt.subplots()

for name in ['y','x']:
    ax.plot(df['time'], df[name], label=name)

ax.set_xlabel("time")
ax.set_ylabel("monthly production")
ax.legend(loc='best')
# plt.show()

df.plot(x='time', y='share')
# plt.show()

# sns.set(color_codes=True)
# ax = sns.regplot(data=df, x='SATV', y='FGPA', marker='+')
# plt.show()

print("\n(b):")
print('(1)')

df_b = df1.iloc[4:]
y = np.array(df_b['dy'])
x = np.array(df_b[['y-1', 'dy-1', 'dy-2', 'dy-3']])

X = sm.add_constant(x)
model = sm.OLS(y, X)
r = model.fit()
print('Linear Regression: dy = b0 + b1*y1 + b2*dy1 + b3*dy2 + b4*dy3 + eps')
print(r.summary())

print('(2)')

y = np.array(df_b['dx'])
x = np.array(df_b[['x-1', 'dx-1', 'dx-2', 'dx-3']])

X = sm.add_constant(x)
model = sm.OLS(y, X)
r = model.fit()
print('Linear Regression: dx = b0 + b1*x1 + b2*dx1 + b3*dx2 + b4*dx3 + eps')
print(r.summary())

print('\n(c):')

y = np.array(df1['y'])
x = np.array(df1['x'])

X = sm.add_constant(x)
model = sm.OLS(y, X)
r = model.fit()
print('Linear Regression: y = b0 + b1*x + eps')
print(r.summary())

df1['yhat_c'] = list(r.fittedvalues)
df1['e_c'] = list(r.resid)
df1['e1_c'] = df1['e_c'].shift(1)
df1['de_c'] = df1['e_c'].diff()
df1['de1_c'] = df1['de_c'].shift(1)
df1['de2_c'] = df1['de_c'].shift(2)
df1['de3_c'] = df1['de_c'].shift(3)

# print(df1.head())
df_c = df1.iloc[4:]

y = np.array(df_c['de_c'])
x = np.array(df_c[['e1_c', 'de1_c', 'de2_c', 'de3_c']])

X = sm.add_constant(x)
model = sm.OLS(y, X)
r = model.fit()
print('Linear Regression: de = b0 + b1*e1 + b2*de1 + b3*de2 + b4*de3 + eps')
print(r.summary())

print('\n(d):')
# print(df1.head())

df_d = df1.iloc[1:]
plot_acf(df_d['dy'], lags=12)

plot_pacf(df_d['dy'], lags=12)
# plt.show()

# print(df1.head(15))
df_d = df1.iloc[13:]

y = np.array(df_d['dy'])
x = np.array(df_d[['dy-1', 'dy-2', 'dy-3', 'dy-4', 'dy-5', 'dy-10', 'dy-12']])

X = sm.add_constant(x)
model = sm.OLS(y, X)
r = model.fit()
print(r.summary())

#########################
print('\n(f1):')

df_f = df.tail(12)
# print(df_f)

y1 = np.array(df_f['dy'])
x1 = np.array(df_f[['dy-1', 'dy-2', 'dy-3', 'dy-4', 'dy-5', 'dy-10', 'dy-12']])

X1 = sm.add_constant(x1)
y1_hat = r.predict(X1)
df_f['dy1_hat'] = list(y1_hat)
RMSE1 = np.sqrt(np.dot(y1-y1_hat, y1-y1_hat)/12.0)
MAE1 = sum(abs(y1-y1_hat))/12.0
print('RMSE1=',RMSE1, 'MAE1=', MAE1)

fig,ax = plt.subplots()

for name in ['dy','dy1_hat']:
    ax.plot(df_f['time'], df_f[name], label=name)

ax.set_xlabel("time")
ax.set_ylabel("dy")
ax.legend(loc='best')

print('\n(e):')

# print(df1.head(15))
df_e = df1.iloc[13:]

y = np.array(df_e['dy'])
x = np.array(df_e[['ecm1', 'dy-1', 'dy-2', 'dy-3', 'dy-4', 'dy-5', 'dy-10', 'dy-12']])

X = sm.add_constant(x)
model = sm.OLS(y, X)
r = model.fit()
print(r.summary())

################################
print('\n(f2):')

y2 = np.array(df_f['dy'])
x2 = np.array(df_f[['ecm1', 'dy-1', 'dy-2', 'dy-3', 'dy-4', 'dy-5', 'dy-10', 'dy-12']])

X2 = sm.add_constant(x2)
y2_hat = r.predict(X2)
df_f['dy2_hat'] = list(y2_hat)
# plt.plot(y2, y2_hat)
# plt.show()
RMSE2 = np.sqrt(np.dot(y2-y2_hat, y2-y2_hat)/12.0)
MAE2 = sum(abs(y2-y2_hat))/12.0
print('RMSE2=',RMSE2, 'MAE2=', MAE2)

fig,ax = plt.subplots()

for name in ['dy','dy2_hat']:
    ax.plot(df_f['time'], df_f[name], label=name)

ax.set_xlabel("time")
ax.set_ylabel("dy")
ax.legend(loc='best')
plt.show()
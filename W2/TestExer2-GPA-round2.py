import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss

df = pd.read_excel('TestExer2-GPA-round2.xls')

print("\n(a):")
print('(i)')

y = np.array(df['FGPA'])
x = np.array(df['SATV'])

# sns.set(color_codes=True)
# ax = sns.regplot(data=df, x='SATV', y='FGPA', marker='+')
# plt.show()

X = sm.add_constant(x)
model = sm.OLS(y, X)
r = model.fit()
print('Linear Regression: FGPA = b0 + b1*SATV + eps')
print('The intercept       = %.3f' % r.params[0])
print('The coeff(SATV)     = %.3f' % r.params[1])
print('The std error(SATV) = %.3f' % r.bse[1])
print('The p-value(SATV)   = %.3f' % r.pvalues[1])

print('(ii)')
b1 = r.params[1]
sb1 = r.bse[1]
print('The 95%% confidence interval for the effect on FGPA of an increase by 1 point in SATV: [%.3f, %.3f]' % (b1-1.96*sb1, b1+1.96*sb1))
print()
print(r.summary())

print('\n(b):')
print('(i)')
y = np.array(df['FGPA'])
x3 = np.array(df[['SATV', 'SATM', 'FEM']])

X3 = sm.add_constant(x3)
model3 = sm.OLS(y, X3)
r3 = model3.fit()
print('Linear Regression: FGPA = b0 + b1*SATV + b2*SATM + b3*FEM + eps')
print('The intercept       = %.3f' % r3.params[0])
print('The coeff(SATV)     = %.3f' % r3.params[1])
print('The std error(SATV) = %.3f' % r3.bse[1])
print('The p-value(SATV)   = %.3f' % r3.pvalues[1])

print('(ii)')
b1 = r3.params[1]
sb1 = r3.bse[1]
print('The 95%% confidence interval for the effect on FGPA of an increase by 1 point in SATV: [%.3f, %.3f]' % (b1-1.96*sb1, b1+1.96*sb1))
print()
print(r3.summary())

print('\n(c):')
df4 = df[['FGPA', 'SATV', 'SATM', 'FEM']]

print(df4.corr())

print('\n(d):')
print('(i)')

g = 1
n = 609
k = 4

print('H0: beta(SATV) = 0 ....... that is, SATV has no effect.')

y = np.array(df['FGPA'])
x1 = np.array(df[['SATM', 'FEM']])

X1 = sm.add_constant(x1)
model1 = sm.OLS(y, X1)
r1 = model1.fit()

R1_squared = r3.rsquared # unrestricted model
R0_squared = r1.rsquared # restricted model

F = ((R1_squared-R0_squared)/g)/((1-R1_squared)/(n-k))
print('g=%d, n=%d, k=%d, n-k=%d' % (g,n,k, n-k))
print('R0_squared(restricted) = %.6f, R1_squared(unrestricted) = %.6f' % (R0_squared, R1_squared))
print('F = ((R1^2 - R0^2)/g) / ((1-R1^2)/(n-k)) = %.6f' % F)

print('At 5%% level, the critical value of F(g,n-k) = F(%d, %d) is %.1f' % (g, n-k, ss.f.ppf(q=1-0.05, dfn=g, dfd=n-k))) # 3.9

print('\n(ii)')
t = r3.tvalues
print('t^2 = %.6f' % (t[1]**2))
print('F = %.6f' % F)

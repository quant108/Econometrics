import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns;
import matplotlib.pyplot as plt

df = pd.read_excel('TestExer1-holiday expenditures-round2.xls')

print("Input Data:")
print(df)
# data = df.values

def calc_stats(x, y):

    n = len(x)

    y1 = y - np.mean(y) # de-meaned y
    x1 = x - np.mean(x) # de-meaned x

    b = np.inner(y1, x1)/np.inner(x1, x1)
    a = np.mean(y) - b*np.mean(x)

    e = y - a - b*x
    s2 = np.inner(e, e)/(n-2)
    s = np.sqrt(s2)
    R2 = 1 - np.inner(e, e)/np.inner(y1, y1)

    # tb = (b-beta)/sb = sum(c*e)/sb # lecture 1.4, slide 9
    c = x1/np.inner(x1, x1)
    beta = b - np.inner(c, e)
    sb2 = s2/np.inner(x1, x1)
    sb = np.sqrt(sb2)
    # tb = (b - beta)/sb
    tb = np.inner(c, e)/sb
    tb1 = b/sb

    print('a = %.4f' % a)
    print('b = %.4f' % b)
    print('standard error = %.4f' % s)
    print('tb = %.4E' % tb)
    print('standard error of b = %.4f' % sb)
    print('t-value of b = %.4f' % tb1)
    # print(sb2, sb, b, beta, np.inner(c, e), sb)

    X = sm.add_constant(x)
    model = sm.OLS(y, X)
    results = model.fit()
    print(results.summary())

print("\nQ1:")

y = np.array(df['Expenditures'])
x = np.array(df['Age'])
calc_stats(x, y)

print('\nQ2:')

sns.set(color_codes=True)
ax = sns.regplot(data=df, x='Age', y='Expenditures', marker='+')
plt.show()

print("\nQ3:")

print('For Age >= 40:')
df_GE40 = df.copy()
df_GE40 = df_GE40[df_GE40['Age']>=40]
y = np.array(df_GE40['Expenditures'])
x = np.array(df_GE40['Age'])
calc_stats(x, y)

print('\nFor Age < 40:')
df_L40 = df.copy()
df_L40 = df_L40[df_L40['Age']<40]
y = np.array(df_L40['Expenditures'])
x = np.array(df_L40['Age'])
calc_stats(x, y)

print("\nQ4:")

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
ax2.set(xlim=(40,60))
sns.regplot(data=df_L40, x='Age', y='Expenditures', marker='+', ax=ax1)
sns.regplot(data=df_GE40, x='Age', y='Expenditures', marker='+', ax=ax2)
plt.show()
